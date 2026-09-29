"""Complete a Phase 0 parent by replaying only its failed native τ-bench review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

from .budget import BudgetSnapshot, RequestBudget
from .communication import observe_communication_protocol
from .manifest import (
    ExperimentManifest,
    role_model_args_for_runtime,
    write_manifest_once,
)
from .phase0 import load_config
from .phase0_run import _load_pinned_tasks, _parse_strategies, _write_json_once
from .tau_adapter import _freeze_tau_evaluator_settings, build_phase0_orchestrator

_PARENT_COMPATIBILITY_FIELDS = (
    "phase",
    "upstream",
    "domain",
    "communication_mode",
    "enforce_communication_protocol",
    "evaluation_type",
    "task_selection",
    "seed",
    "max_steps",
    "max_episodes",
    "request_budget_cap",
    "provider_retries",
    "max_concurrency",
    "real_provider_enabled",
    "role_models",
    "role_model_args",
    "runtime_arguments",
    "source_blob_sha1",
    "customer_strategy_sha256",
    "service_strategy_sha256",
)


def recover_phase0_review(
    config_path: str | Path,
    *,
    source_result_path: str | Path,
    data_dir: str | Path,
) -> dict[str, Any]:
    """Retry the reviewer on a saved, evaluated episode after a reviewer-only outage.

    This creates a new immutable Phase 0 artifact, carries forward the source
    attempt's budget usage and records the exact source trajectory hashes. It
    refuses to re-evaluate a different task, seed, model, or runtime config.
    """

    config = load_config(config_path)
    manifest = ExperimentManifest.from_mapping(config)
    if not manifest.real_provider_enabled:
        raise RuntimeError(
            "review recovery requires an explicitly provider-enabled Phase 0 config"
        )
    data_root = Path(data_dir).expanduser().resolve()
    selection = config["experiment"].get("task_selection", {})
    tasks = _load_pinned_tasks(
        manifest,
        data_dir=data_root,
        task_selection=selection,
        task_ids=(manifest.evolution_task_ids[0],),
    )
    task = tasks[manifest.evolution_task_ids[0]]

    source_path = Path(source_result_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError("source Phase 0 result does not exist")
    source_bytes = source_path.read_bytes()
    source_result = json.loads(source_bytes)
    source_dir = source_path.parent
    source_manifest_path = source_dir / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            "source Phase 0 result is missing its adjacent manifest"
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest_hash = source_manifest.get("manifest_sha256")
    source_manifest_payload = {
        key: value for key, value in source_manifest.items() if key != "manifest_sha256"
    }
    from .manifest import sha256_json

    if (
        not isinstance(source_manifest_hash, str)
        or sha256_json(source_manifest_payload) != source_manifest_hash
    ):
        raise ValueError("source Phase 0 manifest fingerprint is invalid")
    if (
        source_result.get("schema_version") != 1
        or source_result.get("status") != "incomplete"
        or source_result.get("manifest_sha256") != source_manifest_hash
        or source_result.get("experiment_id") != source_manifest.get("experiment_id")
        or source_result.get("native_simulation_saved") is not True
    ):
        raise ValueError(
            "review recovery requires an incomplete result with a saved native simulation"
        )

    current_manifest = manifest.to_document()
    _verify_compatible_parent(source_manifest, current_manifest)
    source_simulation_name = source_result.get("simulation_file")
    if not isinstance(source_simulation_name, str) or not source_simulation_name:
        raise ValueError("source incomplete result has no native simulation reference")
    source_simulation_path = (source_dir / source_simulation_name).resolve()
    if (
        not source_simulation_path.is_relative_to(source_dir)
        or not source_simulation_path.is_file()
    ):
        raise ValueError(
            "source native simulation reference is missing or escapes its run directory"
        )
    source_simulation_bytes = source_simulation_path.read_bytes()
    source_simulation_payload = json.loads(source_simulation_bytes)

    from tau2.data_model.simulation import SimulationRun, UserInfo
    from tau2.evaluator.reviewer import ReviewMode, review_simulation

    simulation = SimulationRun.model_validate(source_simulation_payload)
    reward_info = source_simulation_payload.get("reward_info") or {}
    reward = reward_info.get("reward")
    if (
        str(simulation.task_id) != manifest.evolution_task_ids[0]
        or simulation.seed != manifest.seed
        or not simulation.id
        or simulation.review is not None
        or simulation.auth_classification is not None
        or isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
    ):
        raise ValueError(
            "source trajectory is not the expected evaluated, not-yet-reviewed Phase 0 episode"
        )
    if source_result.get("task_id") is None or str(source_result["task_id"]) != str(
        simulation.task_id
    ):
        raise ValueError("source result and native trajectory task IDs differ")

    try:
        source_budget_data = source_result["provider_budget"]
        source_budget = BudgetSnapshot(**source_budget_data)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source Phase 0 provider budget is malformed") from exc
    if (
        source_budget.cap != manifest.request_budget_cap
        or source_budget.attempts >= source_budget.cap
    ):
        raise ValueError(
            "source Phase 0 budget leaves no room for a bounded review retry"
        )

    output_dir = Path(manifest.output_path)
    if output_dir.exists():
        raise FileExistsError(f"review-recovery output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_manifest_once(output_dir / "manifest.json", manifest)
    output_simulation_path = output_dir / "native-simulation.json"
    simulation_payload = simulation.model_dump(mode="json")
    _write_json_once(output_simulation_path, simulation_payload)
    budget = RequestBudget(manifest.request_budget_cap)
    budget.restore_usage(source_budget)
    source_reference = {
        "source_experiment_id": source_manifest["experiment_id"],
        "source_result_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_manifest_sha256": source_manifest_hash,
        "source_simulation_sha256": hashlib.sha256(source_simulation_bytes).hexdigest(),
        "recovered_stage": "native full review and authentication classification only",
        "source_episode_reward": float(reward),
        "source_failure_type": source_result.get("failure_type"),
        "source_budget": source_budget.to_dict(),
    }
    _write_json_once(output_dir / "review-recovery.json", source_reference)

    try:
        customer, service = _parse_strategies(config["experiment"])
        models = dict(manifest.role_models)
        args = role_model_args_for_runtime(manifest.role_model_args)
        orchestrator = build_phase0_orchestrator(
            task=task,
            agent_model=models["agent"],
            customer_model=models["customer"],
            agent_model_args=args["agent"],
            customer_model_args=args["customer"],
            seed=manifest.seed,
            max_steps=manifest.max_steps,
            customer_strategy=customer,
            service_strategy=service,
            enforce_communication_protocol=manifest.enforce_communication_protocol,
        )
        prompt_hashes = {
            "agent": hashlib.sha256(
                orchestrator.agent.system_prompt.encode("utf-8")
            ).hexdigest(),
            "customer": hashlib.sha256(
                orchestrator.user.system_prompt.encode("utf-8")
            ).hexdigest(),
        }
        if prompt_hashes != source_result.get("rendered_prompt_sha256"):
            raise ValueError(
                "reconstructed frozen prompts differ from the saved native episode"
            )
        from tau2.utils import llm_utils

        with (
            _freeze_tau_evaluator_settings(
                evaluator_model=models["evaluator"],
                evaluator_model_args=args["evaluator"],
                reviewer_model=models["reviewer"],
                reviewer_model_args=args["reviewer"],
            ),
            budget.instrument_tau_llm_utils(llm_utils),
        ):
            review, auth_classification = review_simulation(
                simulation=simulation,
                task=task,
                mode=ReviewMode.FULL,
                user_info=UserInfo(
                    implementation="llm",
                    llm=orchestrator.user.llm,
                    llm_args=orchestrator.user.llm_args,
                    global_simulation_guidelines=orchestrator.user.global_simulation_guidelines,
                    persona_config=orchestrator.user.persona_config,
                ),
                policy=orchestrator.environment.get_policy(),
                review_model=models["reviewer"],
            )
        simulation.review = review
        simulation.auth_classification = auth_classification
        completed_simulation = simulation.model_dump(mode="json")
        _write_json_once(
            output_simulation_path.with_name("reviewed-native-simulation.json"),
            completed_simulation,
        )
        result = {
            "schema_version": 1,
            "status": "complete",
            "experiment_id": manifest.experiment_id,
            "manifest_sha256": manifest.sha256,
            "task_id": str(simulation.task_id),
            "simulation_id": simulation.id,
            "native_reward": float(reward),
            "termination_reason": str(simulation.termination_reason),
            "simulation_file": "reviewed-native-simulation.json",
            "communication_protocol_observation": observe_communication_protocol(
                completed_simulation.get("messages") or (),
                enforcement_enabled=manifest.enforce_communication_protocol,
            ),
            "review": completed_simulation.get("review"),
            "auth_classification": completed_simulation.get("auth_classification"),
            "strategy": {
                "customer": None if customer is None else customer.to_dict(),
                "service": service.to_dict(),
            },
            "rendered_prompt_sha256": prompt_hashes,
            "provider_budget": budget.snapshot().to_dict(),
            "recovered_from": source_reference,
        }
        _write_json_once(output_dir / "phase0-result.json", result)
        return result
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "incomplete",
            "experiment_id": manifest.experiment_id,
            "manifest_sha256": manifest.sha256,
            "task_id": str(simulation.task_id),
            "failure_type": type(exc).__name__,
            "native_simulation_saved": True,
            "simulation_file": output_simulation_path.name,
            "rendered_prompt_sha256": source_result.get("rendered_prompt_sha256"),
            "communication_protocol_observation": source_result.get(
                "communication_protocol_observation"
            ),
            "provider_budget": budget.snapshot().to_dict(),
            "recovered_from": source_reference,
        }
        _write_json_once(output_dir / "phase0-result.json", failure)
        raise


def _verify_compatible_parent(source: dict[str, Any], current: dict[str, Any]) -> None:
    mismatch = [
        field
        for field in _PARENT_COMPATIBILITY_FIELDS
        if source.get(field) != current.get(field)
    ]
    if mismatch:
        raise ValueError(
            f"review-recovery manifest differs from its source in: {', '.join(mismatch)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-phase0-result", type=Path, required=True)
    parser.add_argument("--tau2-data-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = recover_phase0_review(
            args.config,
            source_result_path=args.source_phase0_result,
            data_dir=args.tau2_data_dir,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        KeyError,
        yaml.YAMLError,
    ) as exc:
        print(
            json.dumps({"status": "incomplete", "failure_type": type(exc).__name__}),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # noqa: BLE001 - redact provider exception details from CLI output
        print(
            json.dumps({"status": "incomplete", "failure_type": type(exc).__name__}),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "experiment_id": result["experiment_id"],
                "simulation_id": result["simulation_id"],
                "native_reward": result["native_reward"],
                "provider_budget": result["provider_budget"],
                "recovered_from": result["recovered_from"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
