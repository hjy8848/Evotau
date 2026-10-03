"""Explicit, manifest-bound Phase 0 runner for the pinned τ-bench runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from .budget import RequestBudget
from .communication import observe_communication_protocol
from .eligibility import (
    validate_activation_selection,
    validate_generalization_selection,
    validate_smoke_selection,
)
from .manifest import (
    ActivationSmokeManifest,
    ExperimentManifest,
    verify_git_blob_sha1,
    write_manifest_once,
)
from .phase0 import load_config
from .strategies import CustomerStrategy, ServiceRule, ServiceStrategy
from .tau_adapter import run_phase0_episode, verify_tau2_installation


class Phase0ExecutionError(RuntimeError):
    """A provider execution failed; details are kept out of console output."""


def _load_pinned_task(
    manifest: ExperimentManifest,
    *,
    data_dir: str | Path,
    task_selection: Mapping[str, Any],
) -> Any:
    return _load_pinned_tasks(
        manifest,
        data_dir=data_dir,
        task_selection=task_selection,
        task_ids=(manifest.evolution_task_ids[0],),
    )[manifest.evolution_task_ids[0]]


def _load_pinned_tasks(
    manifest: Any,
    *,
    data_dir: str | Path,
    task_selection: Mapping[str, Any],
    task_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Load requested official-train tasks after verifying every pinned source blob."""

    data_root = Path(data_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError(f"τ-bench data directory does not exist: {data_root}")
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("requested pinned task IDs must be non-empty and unique")
    os.environ["TAU2_DATA_DIR"] = str(data_root)
    verify_tau2_installation()

    import tau2
    from tau2.runner.helpers import get_tasks

    package_root = Path(tau2.__file__).resolve().parent
    for repository_path, digest in manifest.source_blob_sha1:
        if repository_path.startswith("data/"):
            source = data_root / repository_path.removeprefix("data/")
        elif repository_path.startswith("src/tau2/"):
            source = package_root / repository_path.removeprefix("src/tau2/")
        else:
            raise ValueError(f"unsupported pinned source path: {repository_path}")
        verify_git_blob_sha1(source, digest)

    tasks_path = data_root / "tau2/domains/retail/tasks.json"
    split_path = data_root / "tau2/domains/retail/split_tasks.json"
    with tasks_path.open("r", encoding="utf-8") as handle:
        tasks_data = json.load(handle)
    with split_path.open("r", encoding="utf-8") as handle:
        split_data = json.load(handle)
    evolution_ids = getattr(manifest, "evolution_task_ids", None)
    validation_ids = getattr(manifest, "validation_task_ids", None)
    heldout_ids = getattr(manifest, "heldout_task_ids", ())
    if evolution_ids is None:
        evolution_ids = (manifest.evolution_task_id,)
    if validation_ids is None:
        validation_ids = (manifest.validation_task_id,)
    if heldout_ids:
        heldout_split = getattr(manifest, "heldout_split_name", "test")
        validate_generalization_selection(
            tasks_data,
            split_data,
            evolution_task_ids=evolution_ids,
            validation_task_ids=validation_ids,
            heldout_task_ids=heldout_ids,
            excluded_task_ids=task_selection.get("excluded", ()),
        )
        semantic_review_json = getattr(manifest, "task_semantic_review_json", None)
        if semantic_review_json is not None:
            from .task_review import verify_reviewed_task_hashes

            verify_reviewed_task_hashes(json.loads(semantic_review_json), tasks_data)
        train_ids = (*evolution_ids, *validation_ids)
        train_id_set, heldout_id_set = set(train_ids), set(heldout_ids)
        heldout_requested = tuple(task_id for task_id in task_ids if task_id in heldout_id_set)
        train_requested = tuple(task_id for task_id in task_ids if task_id in train_id_set)
        if set(task_ids) - set(train_ids) - set(heldout_ids):
            raise ValueError("requested task IDs are outside the frozen Pilot E/V/H panels")
        selected = []
        if train_requested:
            selected.extend(get_tasks(
                manifest.domain, task_split_name=manifest.split_name,
                task_ids=list(train_requested),
            ))
        if heldout_requested:
            selected.extend(get_tasks(
                manifest.domain, task_split_name=heldout_split,
                task_ids=list(heldout_requested),
            ))
    elif isinstance(manifest, ActivationSmokeManifest):
        panel = validate_activation_selection(
            tasks_data,
            split_data,
            evolution_task_ids=manifest.evolution_task_ids,
            validation_task_ids=manifest.validation_task_ids,
            excluded_task_ids=manifest.excluded_task_ids,
        )
        review_document = json.loads(manifest.task_semantic_review_json)
        from .task_review import verify_reviewed_task_hashes

        verify_reviewed_task_hashes(review_document, tasks_data)
        expected = panel.evolution_task_ids + panel.validation_task_ids
        if set(task_ids) != set(expected):
            raise ValueError("activation runner must load exactly its frozen E/V task panel")
        selected = get_tasks(
            manifest.domain,
            task_split_name=manifest.split_name,
            task_ids=list(task_ids),
        )
    else:
        validate_smoke_selection(
            tasks_data,
            split_data,
            evolution_task_id=evolution_ids[0],
            validation_task_id=validation_ids[0],
            excluded_task_ids=task_selection.get("excluded", ()),
        )
        selected = get_tasks(
            manifest.domain,
            task_split_name=manifest.split_name,
            task_ids=list(task_ids),
        )
    tasks_by_id = {str(task.id): task for task in selected}
    if set(tasks_by_id) != set(task_ids):
        raise ValueError("pinned τ-bench did not return exactly the requested official-train tasks")
    return tasks_by_id


def _parse_strategies(
    experiment: Mapping[str, Any],
) -> tuple[Any | None, ServiceStrategy]:
    customer_data = experiment.get("customer_strategy")
    if customer_data is None:
        customer = None
    elif isinstance(customer_data, Mapping) and customer_data.get("schema_version") == 3:
        from .customer_skill_v3 import validate_skill

        if set(customer_data) != {"schema_version", "skill"} or not isinstance(customer_data["skill"], Mapping):
            raise ValueError("V3 Customer baseline has an invalid explicit schema")
        raw_skill = customer_data["skill"]
        normalized = validate_skill(
            raw_skill,
            incumbent=CustomerStrategy.v2_baseline(),
            operation="create",
            verified_failure_ids=raw_skill.get("evidence_refs", ()),
        )
        if normalized.to_dict() != customer_data:
            raise ValueError("V3 Customer baseline must be canonical")
        customer = normalized
    else:
        customer = CustomerStrategy(**customer_data)
    service_data = experiment.get("service_strategy") or {"rules": []}
    rules = tuple(
        ServiceRule(
            rule_id=item["rule_id"],
            policy_ref=item["policy_ref"],
            trigger=item["trigger"],
            required_execution=item["required_execution"],
            evidence_refs=tuple(item["evidence_refs"]),
        )
        for item in service_data.get("rules", ())
    )
    return customer, ServiceStrategy(rules)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def execute_phase0(
    *,
    manifest: ExperimentManifest,
    config: Mapping[str, Any],
    task: Any,
    episode_runner: Callable[..., tuple[Any, RequestBudget]] | None = None,
    service_token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Run one explicitly enabled episode and save an immutable result record."""

    if ExperimentManifest.from_mapping(config).sha256 != manifest.sha256:
        raise ValueError("run configuration does not match the frozen Phase 0 manifest")
    if not manifest.real_provider_enabled:
        raise RuntimeError(
            "real provider use is disabled in the Phase 0 manifest; enable it and freeze all role models"
        )
    if task.id != manifest.evolution_task_ids[0]:
        raise ValueError("loaded task does not match the frozen evolution task")
    customer_strategy, service_strategy = _parse_strategies(config["experiment"])
    if service_strategy.rules and service_token_counter is None:
        raise ValueError("non-empty ServiceStrategy requires a model-matched token counter")

    output_dir = Path(manifest.output_path)
    manifest_path = output_dir / "manifest.json"
    simulation_path = output_dir / "native-simulation.json"
    result_path = output_dir / "phase0-result.json"
    if manifest_path.exists() or simulation_path.exists() or result_path.exists():
        raise FileExistsError(f"Phase 0 output already exists: {output_dir}")

    write_manifest_once(manifest_path, manifest)
    budget = RequestBudget(manifest.request_budget_cap)
    runner = episode_runner or run_phase0_episode
    simulation_saved = False
    simulation_payload: dict[str, Any] | None = None
    prompt_hashes: dict[str, str] = {}

    def record_prompt_hashes(orchestrator: Any) -> None:
        prompt_hashes["agent"] = hashlib.sha256(
            orchestrator.agent.system_prompt.encode("utf-8")
        ).hexdigest()
        prompt_hashes["customer"] = hashlib.sha256(
            orchestrator.user.system_prompt.encode("utf-8")
        ).hexdigest()

    def persist_native_simulation(simulation: Any) -> None:
        nonlocal simulation_saved, simulation_payload
        simulation_payload = simulation.model_dump(mode="json")
        _write_json_once(simulation_path, simulation_payload)
        simulation_saved = True

    try:
        simulation, _ = runner(
            manifest=manifest,
            task=task,
            customer_strategy=customer_strategy,
            service_strategy=service_strategy,
            service_token_counter=service_token_counter,
            request_budget=budget,
            on_orchestrator=record_prompt_hashes,
            on_simulation=persist_native_simulation,
        )
        simulation_payload = simulation.model_dump(mode="json")
        if not simulation_saved:
            persist_native_simulation(simulation)
        reward_info = simulation_payload.get("reward_info") or {}
        result = {
            "schema_version": 1,
            "status": "complete",
            "experiment_id": manifest.experiment_id,
            "manifest_sha256": manifest.sha256,
            "task_id": simulation.task_id,
            "simulation_id": simulation.id,
            "native_reward": reward_info.get("reward"),
            "termination_reason": simulation_payload.get("termination_reason"),
            "simulation_file": simulation_path.name,
            "communication_protocol_observation": observe_communication_protocol(
                simulation_payload.get("messages") or (),
                enforcement_enabled=manifest.enforce_communication_protocol,
            ),
            "review": simulation_payload.get("review"),
            "auth_classification": simulation_payload.get("auth_classification"),
            "strategy": {
                "customer": None if customer_strategy is None else customer_strategy.to_dict(),
                "service": service_strategy.to_dict(),
            },
            "rendered_prompt_sha256": prompt_hashes,
            "provider_budget": budget.snapshot().to_dict(),
        }
    except Exception as exc:
        failure_record = {
            "schema_version": 1,
            "status": "incomplete",
            "experiment_id": manifest.experiment_id,
            "manifest_sha256": manifest.sha256,
            "task_id": task.id,
            "failure_type": type(exc).__name__,
            "rendered_prompt_sha256": prompt_hashes,
            "native_simulation_saved": simulation_saved,
            "simulation_file": simulation_path.name if simulation_saved else None,
            "communication_protocol_observation": (
                None if simulation_payload is None else observe_communication_protocol(
                    simulation_payload.get("messages") or (),
                    enforcement_enabled=manifest.enforce_communication_protocol,
                )
            ),
            "provider_budget": budget.snapshot().to_dict(),
        }
        _write_json_once(result_path, failure_record)
        raise Phase0ExecutionError(type(exc).__name__) from exc

    _write_json_once(result_path, result)
    return result


def run_from_config(
    config_path: str | Path,
    *,
    tau2_data_dir: str | Path | None = None,
    episode_runner: Callable[..., tuple[Any, RequestBudget]] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = ExperimentManifest.from_mapping(config)
    if not manifest.real_provider_enabled:
        raise RuntimeError(
            "real provider use is disabled in the Phase 0 manifest; enable it and freeze all role models"
        )
    data_dir = tau2_data_dir or os.environ.get("TAU2_DATA_DIR")
    if data_dir is None:
        raise ValueError("provide --tau2-data-dir or set TAU2_DATA_DIR to the pinned data directory")
    task = _load_pinned_task(
        manifest,
        data_dir=data_dir,
        task_selection=config["experiment"]["task_selection"],
    )
    token_counter = None
    if (config["experiment"].get("service_strategy") or {}).get("rules"):
        from litellm import token_counter as count_tokens

        agent_model = dict(manifest.role_models)["agent"]
        token_counter = lambda text: count_tokens(model=agent_model, text=text)
    return execute_phase0(
        manifest=manifest,
        config=config,
        task=task,
        episode_runner=episode_runner,
        service_token_counter=token_counter,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.yaml"))
    parser.add_argument(
        "--tau2-data-dir",
        type=Path,
        help="path to data/ from the pinned tau-bench checkout (or set TAU2_DATA_DIR)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_from_config(args.config, tau2_data_dir=args.tau2_data_dir)
    except Phase0ExecutionError as exc:
        print(f"Phase 0 run failed ({exc}); inspect the immutable run record", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, KeyError, yaml.YAMLError) as exc:
        print(f"Phase 0 run failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI must safely report all SDK exception types
        print(
            f"Phase 0 run failed ({type(exc).__name__}); inspect the immutable run record",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "experiment_id": result["experiment_id"],
                "manifest_sha256": result["manifest_sha256"],
                "task_id": result["task_id"],
                "simulation_id": result.get("simulation_id"),
                "native_reward": result.get("native_reward"),
                "provider_budget": result["provider_budget"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
