from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest

from evotau.activation_smoke_run import run_from_config
from evotau.eligibility import validate_activation_selection
from evotau.lifecycle import TwoGenerationSmoke
from evotau.manifest import ActivationSmokeManifest
from evotau.phase0 import load_config

ROOT = Path(__file__).resolve().parents[1]


def reviewed_panel() -> dict:
    task_ids = tuple(f"synthetic-e-{index:02d}" for index in range(10)) + ("93",)
    panels = tuple((task_id, "evolution") for task_id in task_ids[:10]) + (("93", "validation"),)
    return {
        "schema_version": 1,
        "review_id": "synthetic-activation-review",
        "reviewer_id": "synthetic-test-fixture",
        "reviewed_at": "2026-09-29T00:00:00Z",
        "task_reviews": [
            {
                "task_id": task_id,
                "panel": panel,
                "task_sha256": "a" * 64,
                "no_deception_required": True,
                "policy_tool_compatible": True,
                "satisfiable": True,
                "rationale": "Synthetic unit-test fixture; not a research task judgment.",
            }
            for task_id, panel in panels
        ],
        "pairwise_reviews": [
            {
                "left_task_id": left,
                "right_task_id": right,
                "distinct_scenario": True,
                "rationale": "Synthetic unit-test fixture only.",
            }
            for left, right in combinations(task_ids, 2)
        ],
    }


def activation_config() -> dict:
    return load_config(ROOT / "configs/activation-smoke-inferai-deepseek-v4-flash.yaml")


def test_activation_manifest_freezes_independent_e10_g1_seed1_design() -> None:
    manifest = ActivationSmokeManifest.from_mapping(
        activation_config(), task_review_document=reviewed_panel(),
    )

    assert len(manifest.evolution_task_ids) == 10
    assert manifest.validation_task_ids == ("93",)
    assert manifest.seed == 1
    assert manifest.generations == 1
    assert manifest.customer_candidates == 2
    assert manifest.max_concurrency == 4
    assert manifest.max_episodes == 100
    assert manifest.request_budget_cap == 1800
    payload = manifest.to_payload()
    assert payload["task_selection"]["heldout"] == []
    assert payload["task_selection"]["evolution"] == list(manifest.evolution_task_ids)
    assert payload["task_semantic_review_sha256"] == manifest.task_semantic_review_sha256


def test_uncapped_request_budget_is_only_enabled_for_explicit_diagnostic_config() -> None:
    config = activation_config()
    config["experiment"]["request_budget_cap"] = None
    config["experiment"]["unbounded_provider_budget"] = True
    for args in config["experiment"]["model_args"].values():
        args.pop("max_tokens", None)
    manifest = ActivationSmokeManifest.from_mapping(
        config, task_review_document=reviewed_panel(),
    )
    assert manifest.request_budget_cap is None
    assert manifest.unbounded_provider_budget is True
    assert all("max_tokens" not in args for _, args in manifest.role_model_args)
    assert manifest.to_payload()["unbounded_provider_budget"] is True

    config["experiment"]["unbounded_provider_budget"] = False
    with pytest.raises(ValueError, match="requires explicit configuration"):
        ActivationSmokeManifest.from_mapping(config, task_review_document=reviewed_panel())


def test_activation_requires_exactly_ten_human_reviewed_e_tasks() -> None:
    review = reviewed_panel()
    review["task_reviews"] = review["task_reviews"][1:]
    review["pairwise_reviews"] = [
        row for row in review["pairwise_reviews"]
        if row["left_task_id"] != "synthetic-e-00" and row["right_task_id"] != "synthetic-e-00"
    ]
    with pytest.raises(ValueError, match="exactly ten unique E tasks"):
        ActivationSmokeManifest.from_mapping(activation_config(), task_review_document=review)


def test_activation_selection_uses_only_selected_train_tasks_and_leaves_test_unselected() -> None:
    evolution_ids = tuple(f"E{index}" for index in range(10))
    validation_ids = ("V93",)
    selected_ids = evolution_ids + validation_ids
    tasks = [
        {
            "id": task_id,
            "user_scenario": {
                "instructions": {
                    "task_instructions": "You want to update a pending order.",
                    "known_info": f"Contact you at user{index}@example.com; order W{index:07d}.",
                },
            },
            "evaluation_criteria": {"actions": [{
                "name": "modify_pending_order_address",
                "arguments": {"order_id": f"W{index:07d}"},
            }]},
        }
        for index, task_id in enumerate(selected_ids)
    ]
    tasks.append({"id": "H-sealed-fixture", "user_scenario": {"instructions": "opaque"}})
    split = {"train": list(selected_ids), "test": ["H-sealed-fixture"]}

    selection = validate_activation_selection(
        tasks,
        split,
        evolution_task_ids=evolution_ids,
        validation_task_ids=validation_ids,
    )

    assert selection.evolution_task_ids == evolution_ids
    assert selection.validation_task_ids == validation_ids
    assert selection.heldout_task_ids == ()


def test_activation_refuses_filled_config_task_list_and_any_heldout_panel() -> None:
    config = activation_config()
    config["experiment"]["task_selection"]["evolution"] = ["unreviewed-id"]
    with pytest.raises(ValueError, match="come only from the attached human-reviewed task panel"):
        ActivationSmokeManifest.from_mapping(config, task_review_document=reviewed_panel())

    config = activation_config()
    config["experiment"]["task_selection"]["heldout"] = ["unopened-id"]
    with pytest.raises(ValueError, match="keep H empty"):
        ActivationSmokeManifest.from_mapping(config, task_review_document=reviewed_panel())


def test_generic_generation_controller_accepts_activation_single_generation_without_running() -> None:
    manifest = {
        "generations": 1,
        "max_episodes": 100,
        "max_concurrency": 4,
    }
    controller = TwoGenerationSmoke(
        manifest=manifest,
        checkpoint_path="unused-by-constructor",
        runner=lambda **_kwargs: None,
        task_ids=("synthetic-e-00", "93"),
        evolution_task_ids=tuple(f"synthetic-e-{index:02d}" for index in range(10)),
        seed=1,
        generations=1,
        max_episodes=100,
    )

    assert controller.generations == 1
    assert controller.max_concurrency == 4
    assert controller.episode_attempts == 0


def test_activation_missing_review_stops_before_data_or_provider_loading(tmp_path: Path) -> None:
    import yaml

    config = activation_config()
    config["experiment"]["task_review_path"] = "missing-human-review.json"
    config_path = tmp_path / "activation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="existing task-review workflow"):
        run_from_config(
            config_path,
            tau2_data_dir=tmp_path / "missing-pinned-data",
            provider_plugin="must_not_import:factory",
        )
