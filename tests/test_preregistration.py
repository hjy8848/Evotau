from __future__ import annotations

import hashlib
import json

import pytest

from evotau.preregistration import main, validate_formal_preregistration


def sha(index: str) -> str:
    return hashlib.sha256(index.encode()).hexdigest()


def formal_plan():
    primary = [
        ("RQ1", "verified_task_signature_yield", "static_customer", "greater"),
        ("RQ1", "verified_task_signature_yield", "random_mutation", "greater"),
        ("RQ2", "target_failure_rate_reduction", "incumbent_service", "greater"),
        ("RQ2", "clean_success_rate_change", "incumbent_service", "non_inferior"),
        ("RQ2", "heldout_success_rate_change", "incumbent_service", "non_inferior"),
        ("RQ3", "sustained_two_chain_response", "frozen_service", "greater"),
        ("RQ3", "sustained_two_chain_response", "random_mutation", "greater"),
    ]
    hypotheses = [
        {
            "hypothesis_id": f"h-{index}",
            "research_question": rq,
            "endpoint": endpoint,
            "contrast": contrast,
            "alternative": alternative,
            "primary": True,
            "multiplicity_family": f"family-{rq}",
        }
        for index, (rq, endpoint, contrast, alternative) in enumerate(primary)
    ]
    pilot_artifacts = {
        name: {"path": f"pilot/{name}.json", "sha256": sha(name)}
        for name in ("cost_profile", "rq1_report", "rq2_report", "rq3_report")
    }
    return {
        "schema_version": 1,
        "study_id": "evotau-formal-test",
        "registry_url": "https://osf.io/abcd1",
        "registered_at_utc": "2026-09-29T00:00:00Z",
        "evo_tau_commit": "a" * 40,
        "tau2_commit": "b" * 40,
        "shared_manifest_artifact": {"path": "manifest.json", "sha256": sha("manifest")},
        "pilot_artifacts": pilot_artifacts,
        "model_ids": {role: f"provider/{role}-model" for role in ("agent", "customer", "reviewer")},
        "sampling_parameters": {
            role: {"temperature": 0.0, "max_tokens": 1000}
            for role in ("agent", "customer", "reviewer")
        },
        "task_panels": {"E": ["E1", "E2"], "V": ["V1"], "H": ["H1"]},
        "eligibility_review_artifact": {"path": "eligibility-review.json", "sha256": sha("eligibility")},
        "independent_evolution_seeds": [17, 29, 43, 59, 71, 83],
        "condition_budget_caps": {
            name: 12000 for name in (
                "adaptive_customer", "static_customer", "random_mutation",
                "adaptive_coevolution", "frozen_service", "frozen_customer",
                "no_historical_replay", "one_shot_repair",
            )
        },
        "hypotheses": hypotheses,
        "power_calculations": [
            {
                "hypothesis_id": item["hypothesis_id"],
                "pilot_artifact_sha256": pilot_artifacts[
                    {"RQ1": "rq1_report", "RQ2": "rq2_report", "RQ3": "rq3_report"}
                    [item["research_question"]]
                ]["sha256"],
                "calculator_artifact": {
                    "path": f"calculators/{item['hypothesis_id']}.py",
                    "sha256": sha("calculator-" + item["hypothesis_id"]),
                },
                "calculation_artifact": {
                    "path": f"power/{item['hypothesis_id']}.json",
                    "sha256": sha("power-" + item["hypothesis_id"]),
                },
                "target_power": 0.8,
                "planned_seed_blocks": 6,
            }
            for item in hypotheses
        ],
        "familywise_alpha": 0.05,
        "multiple_comparison_method": "holm",
        "noninferiority_margins": {
            "clean_success_rate_change": 0.05,
            "heldout_success_rate_change": 0.05,
        },
        "ablations": [
            "frozen_customer", "frozen_service", "no_historical_replay",
            "random_instead_of_failure_conditioned_mutation",
        ],
        "human_review_plan": {
            "reviewer_count": 2,
            "calibration_candidate_count": 30,
            "sample_seed": 101,
            "blind_conditions": True,
        },
        "exclusion_rules": ["exclude infrastructure errors before unblinding"],
        "missing_data_rule": "retain missing denominators and report incomplete endpoints",
        "stopping_rule": "stop at the frozen run and request caps",
        "blinded_before_outcome_access": True,
    }


def test_formal_preregistration_requires_pilot_backed_complete_research_design():
    document = formal_plan()
    raw = json.dumps(document, sort_keys=True).encode()
    summary = validate_formal_preregistration(
        document, exact_input_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert summary.status == "structurally_valid_registry_reference_unverified"
    assert summary.independent_seed_blocks == 6
    assert summary.primary_hypotheses == 7
    assert dict(summary.task_counts) == {"E": 2, "H": 1, "V": 1}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["condition_budget_caps"].update({"static_customer": 11999}), "same frozen"),
        (lambda value: value["task_panels"]["H"].append("E1"), "must be disjoint"),
        (lambda value: value["independent_evolution_seeds"].pop(), "largest preregistered sample size"),
        (lambda value: value["noninferiority_margins"].pop("heldout_success_rate_change"), "heldout_success"),
        (lambda value: value.update({"blinded_before_outcome_access": False}), "blinded_before_outcome_access"),
        (lambda value: value["pilot_artifacts"].update({"rq3_report": {
            "path": "pilot/rq3_report.json", "sha256": "not-a-hash",
        }}), "lowercase SHA-256"),
    ],
)
def test_formal_preregistration_fails_closed_on_unfrozen_inputs(mutate, message):
    document = formal_plan()
    mutate(document)
    with pytest.raises(ValueError, match=message):
        validate_formal_preregistration(document, exact_input_sha256=sha("plan"))


def test_formal_preregistration_rejects_missing_required_primary_contrast():
    document = formal_plan()
    document["hypotheses"] = [
        item for item in document["hypotheses"]
        if not (item["research_question"] == "RQ3" and item["contrast"] == "random_mutation")
    ]
    document["power_calculations"] = [
        item for item in document["power_calculations"]
        if item["hypothesis_id"] in {hypothesis["hypothesis_id"] for hypothesis in document["hypotheses"]}
    ]
    with pytest.raises(ValueError, match="missing planned RQ contrasts"):
        validate_formal_preregistration(document, exact_input_sha256=sha("plan"))


def test_formal_preflight_verifies_local_artifact_hashes_and_writes_once(tmp_path, capsys):
    document = formal_plan()

    def write_ref(reference, content):
        path = tmp_path / reference["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = content.encode()
        path.write_bytes(raw)
        reference["sha256"] = hashlib.sha256(raw).hexdigest()

    write_ref(document["shared_manifest_artifact"], "manifest")
    write_ref(document["eligibility_review_artifact"], "eligibility")
    for name, reference in document["pilot_artifacts"].items():
        write_ref(reference, name)
    for item in document["power_calculations"]:
        write_ref(item["calculation_artifact"], "power-" + item["hypothesis_id"])
        write_ref(item["calculator_artifact"], "calculator-" + item["hypothesis_id"])
        rq = next(row["research_question"] for row in document["hypotheses"]
                  if row["hypothesis_id"] == item["hypothesis_id"])
        pilot_name = {"RQ1": "rq1_report", "RQ2": "rq2_report", "RQ3": "rq3_report"}[rq]
        item["pilot_artifact_sha256"] = document["pilot_artifacts"][pilot_name]["sha256"]

    plan_path = tmp_path / "formal-plan.json"
    plan_path.write_text(json.dumps(document, sort_keys=True, indent=2), encoding="utf-8")
    output_path = tmp_path / "formal-preflight.json"
    assert main(["--input", str(plan_path), "--output", str(output_path)]) == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["local_artifacts_verified"] is True
    assert report["registry_url"] == document["registry_url"]

    with pytest.raises(SystemExit) as error:
        main(["--input", str(plan_path), "--output", str(output_path)])
    assert error.value.code == 2
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


def test_formal_preflight_rejects_artifact_path_escape(tmp_path):
    document = formal_plan()
    document["pilot_artifacts"]["rq1_report"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="must stay within"):
        validate_formal_preregistration(document, exact_input_sha256=sha("plan"))
