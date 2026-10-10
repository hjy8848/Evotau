"""Evidence-bound Analyst contracts and auditable semantic diversity selection."""

from copy import deepcopy
from itertools import combinations

from .evolution_candidates import EvolverSchemaError, validate_direct_evidence
from .evolver_recovery import RecoveryExhausted
from .provider_diagnostics import safe_error

ANALYST_VERSION = "analyst_skill_v_validation_v3"
FIELDS = {
    "mechanism_id",
    "target_task_ids",
    "protected_success_task_ids",
    "observed_deviation",
    "root_cause_hypothesis",
    "evidence_refs",
    "expected_behavior_change",
    "alternative_explanations",
    "regression_risk",
    "repairability",
}
TEXT_FIELDS = {
    "mechanism_id",
    "observed_deviation",
    "root_cause_hypothesis",
    "expected_behavior_change",
    "regression_risk",
}


def validate_analysis(result, context):
    if (
        not isinstance(result, dict)
        or set(result) != {"hypotheses", "insufficient_evidence_reason"}
        or not isinstance(result["hypotheses"], list)
        or not isinstance(result["insufficient_evidence_reason"], str)
    ):
        raise ValueError("Invalid Analyst response schema")
    if len(result["hypotheses"]) > context["requested_hypotheses"]:
        raise ValueError("Analyst exceeded requested hypothesis count")
    ids = set()
    for h in result["hypotheses"]:
        if not isinstance(h, dict) or set(h) != FIELDS:
            raise ValueError(
                "Invalid Analyst hypothesis fields; no Skill payload allowed"
            )
        if any(not isinstance(h[k], str) or not h[k].strip() for k in TEXT_FIELDS):
            raise ValueError("Analyst hypothesis requires explicit nonempty text")
        if h["mechanism_id"] in ids:
            raise ValueError("Duplicate mechanism_id")
        if any(c in h["mechanism_id"] for c in ("/", "\\", "\x00")):
            raise ValueError("Unsafe mechanism_id")
        ids.add(h["mechanism_id"])
        if h["repairability"] not in ("skill", "not_skill", "uncertain"):
            raise ValueError("Invalid repairability enum")
        for key in (
            "target_task_ids",
            "protected_success_task_ids",
            "alternative_explanations",
        ):
            if (
                not isinstance(h[key], list)
                or any(not isinstance(v, str) or not v.strip() for v in h[key])
                or len(set(h[key])) != len(h[key])
            ):
                raise ValueError("Invalid hypothesis array")
        if not isinstance(h["evidence_refs"], list):
            raise TypeError("Hypothesis evidence_refs must be array")
        # Reuse exact retained-message registry and task×seed outcome validation.
        validate_direct_evidence(
            {
                "operation": "add",
                "evidence_task_ids": h["target_task_ids"],
                "protected_success_task_ids": h["protected_success_task_ids"],
                "evidence_refs": h["evidence_refs"],
            },
            context,
        )
    if not result["hypotheses"] and not result["insufficient_evidence_reason"].strip():
        raise ValueError("Empty Analyst output requires an insufficiency reason")
    return result


def analysis_outcome(callback, context):
    try:
        result = callback()
    except RecoveryExhausted as error:
        return {"status": "REJECTED", "error_type": "RECOVERY_EXHAUSTED",
                "reason": str(error), "recovery": error.outcome, "hypotheses": []}
    except EvolverSchemaError as error:
        return {
            "status": "REJECTED",
            "error_type": "EvolverSchemaError",
            "reason": safe_error(error),
            "diagnostics_ref": getattr(error, "diagnostics_ref", None),
            "hypotheses": [],
        }
    # Do not catch provider/JSON/runtime exceptions as ordinary rejection.
    try:
        validate_analysis(result, context)
    except (ValueError, TypeError, KeyError) as error:
        return {
            "status": "REJECTED",
            "error_type": "AnalystContractError",
            "reason": safe_error(error),
            "raw_analysis": result,
            "hypotheses": [],
        }
    return {"status": "VALID", **result}


def validate_assigned_mutation(mutation, context):
    h = context.get("assigned_hypothesis")
    if h is None or mutation["operation"] == "no_op":
        return mutation
    if (
        mutation["target_cluster_id"] != h["mechanism_id"]
        or mutation["root_cause_hypothesis"] != h["root_cause_hypothesis"]
        or set(mutation["evidence_task_ids"]) != set(h["target_task_ids"])
        or not set(mutation["protected_success_task_ids"])
        <= set(h["protected_success_task_ids"])
    ):
        raise ValueError("Mutation does not match assigned hypothesis scope")
    if any(ref not in h["evidence_refs"] for ref in mutation["evidence_refs"]):
        raise ValueError("Mutation cites evidence outside assigned hypothesis")
    return mutation


def validate_dedup_review(review, hypotheses):
    if (
        not isinstance(review, dict)
        or set(review) != {"comparisons"}
        or not isinstance(review["comparisons"], list)
    ):
        raise ValueError("Invalid semantic diversity review")
    ids = [h["mechanism_id"] for h in hypotheses]
    expected = {frozenset(pair) for pair in combinations(ids, 2)}
    seen = set()
    for c in review["comparisons"]:
        if not isinstance(c, dict) or set(c) != {
            "left",
            "right",
            "same_mechanism",
            "reason",
        }:
            raise ValueError("Invalid semantic comparison fields")
        if not isinstance(c["left"], str) or not isinstance(c["right"], str):
            raise TypeError("Invalid comparison IDs")
        pair = frozenset((c["left"], c["right"]))
        if pair not in expected or pair in seen:
            raise ValueError("Invalid/duplicate hypothesis pair")
        if (
            (c["same_mechanism"] is not None and type(c["same_mechanism"]) is not bool)
            or not isinstance(c["reason"], str)
            or not c["reason"].strip()
        ):
            raise ValueError("Semantic comparison requires explicit uncertainty/reason")
        seen.add(pair)
    if seen != expected:
        raise ValueError("Missing semantic pair comparison")
    return review


def select_distinct_hypotheses(hypotheses, review, count):
    """LLM semantic judgments are attributed, not passed off as deterministic proof.

    Exact operational text equality is deterministic; uncertain pairs are conservatively
    not both allocated. Skill-repairable cases precede uncertain cases; not_skill is retained
    in the Analyst artifact but not mutated. Stable evidence count/input order break ties.
    """
    validate_dedup_review(review, hypotheses)
    relations = {frozenset((c["left"], c["right"])): c for c in review["comparisons"]}
    eligible = [h for h in hypotheses if h["repairability"] != "not_skill"]
    eligible.sort(
        key=lambda h: (h["repairability"] != "skill", -len(h["evidence_refs"]))
    )
    selected = []
    exclusions = [
        {
            "mechanism_id": h["mechanism_id"],
            "status": "not_skill",
            "reason": h["expected_behavior_change"],
            "source": "analyst_repairability_judgment",
        }
        for h in hypotheses
        if h["repairability"] == "not_skill"
    ]

    def normalize(text):
        return " ".join(text.casefold().split())

    for h in eligible:
        conflict = None
        for previous in selected:
            c = relations[frozenset((h["mechanism_id"], previous["mechanism_id"]))]
            exact = normalize(h["expected_behavior_change"]) == normalize(
                previous["expected_behavior_change"]
            )
            if exact or c["same_mechanism"] is not False:
                conflict = {
                    "mechanism_id": h["mechanism_id"],
                    "relative_to": previous["mechanism_id"],
                    "status": "exact_duplicate"
                    if exact
                    else "semantic_duplicate"
                    if c["same_mechanism"]
                    else "uncertain_deferred",
                    "reason": c["reason"],
                    "source": "deterministic_text_match"
                    if exact
                    else "model_semantic_judgment",
                }
                break
        if conflict:
            exclusions.append(conflict)
        elif len(selected) < count:
            selected.append(deepcopy(h))
    return {
        "assigned_hypotheses": selected,
        "excluded_hypotheses": exclusions,
        "semantic_distinctness_proven": False,
        "selection_order": "repairability, retained evidence count, stable input order",
    }


def diversity_outcome(callback, hypotheses):
    try:
        review = callback()
    except RecoveryExhausted as error:
        return {"stage": "service_mechanism_diversity", "status": "DEGRADED",
                "reason": "dedup_json_or_schema_recovery_exhausted",
                "fallback": "single_validated_hypothesis", "recovery": error.outcome,
                "recovery_attempts": error.outcome["recovery_attempts"]}
    except EvolverSchemaError as error:
        return {
            "status": "REJECTED",
            "error_type": "EvolverSchemaError",
            "reason": safe_error(error),
            "diagnostics_ref": getattr(error, "diagnostics_ref", None),
        }
    try:
        validate_dedup_review(review, hypotheses)
    except (ValueError, TypeError, KeyError) as error:
        return {
            "status": "REJECTED",
            "error_type": "DiversityContractError",
            "reason": safe_error(error),
            "raw_review": review,
        }
    return {"status": "VALID", **review}


def conservative_dedup_fallback(hypotheses, outcome):
    """Only previously evidence-validated skill hypotheses; stable one-candidate fallback."""
    eligible = [h for h in hypotheses if h["repairability"] == "skill"]
    eligible.sort(key=lambda h: -len(h["evidence_refs"]))  # Stable source order ties.
    assigned = deepcopy(eligible[:1])
    ids = {h["mechanism_id"] for h in assigned}
    return {"status": "DEGRADED", "reason": outcome["reason"],
            "fallback": "single_validated_hypothesis" if assigned else "NO_OP",
            "assigned_count": len(assigned), "assigned_hypotheses": assigned,
            "excluded_hypotheses": [{"mechanism_id": h["mechanism_id"],
                "status": "DEFERRED_DUE_TO_DEDUP_FAILURE"} for h in hypotheses
                if h["mechanism_id"] not in ids],
            "semantic_distinctness_proven": False,
            "selection_order": "skill only, retained evidence count, stable input order"}
