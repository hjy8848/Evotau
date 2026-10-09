"""Small run manifest for the alternating Customer/Service evolution method."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .tau_provenance import (
    DEFAULT_ROLE_MODEL_ARGS,
    EVOLUTION_ROLE_NAMES,
    TAU2_PACKAGE_VERSION,
    TAU_BENCH_COMMIT,
    TAU_BENCH_REPOSITORY,
    _freeze_role_models,
    _relative_path,
    _role_model_args_payload,
    alternating_source_paths,
    capture_code_provenance,
    freeze_role_model_args,
    sha256_json,
)

DEFAULT_MAX_PARALLEL_EPISODES = 4
MAX_PARALLEL_EPISODES = 8


@dataclass(frozen=True, slots=True)
class AlternatingManifest:
    """Only the frozen inputs needed to reproduce an alternating run."""

    experiment_id: str
    upstream_repository: str
    upstream_commit: str
    upstream_package_version: str
    evotau_git_commit: str | None
    evotau_working_tree_clean: bool | None
    evotau_source_sha256: str
    evolution_task_ids: tuple[str, ...]
    validation_task_ids: tuple[str, ...]
    heldout_task_ids: tuple[str, ...]
    excluded_task_ids: tuple[str, ...]
    seed: int
    generations: int
    customer_candidates: int
    clean_panel_size: int
    max_steps: int
    max_parallel_episodes: int
    request_budget_cap: int | None
    real_provider_enabled: bool
    role_models: tuple[tuple[str, str | None], ...]
    role_model_args: tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...]
    source_blob_sha1: tuple[tuple[str, str], ...]
    initial_customer_strategy: str
    initial_service_strategy: str
    output_path: str
    checkpoint_path: str
    evolution_fitness_seed: int | None = None
    run_validation: bool = True
    run_heldout: bool = True
    domain: str = "retail"
    split_name: str = "train"
    heldout_split_name: str = "test"
    enforce_communication_protocol: bool = False
    evaluation_type: str = "all"
    source_scope: str = "runtime-v2"
    customer_carrier: str = "prompt_strategy"
    service_carrier: str = "prompt_strategy"
    service_skill_runtime: str = "inject_all"
    service_mutation_ops: tuple[str, ...] = ("add", "update", "no_op")
    max_service_mutations_per_generation: int = 1
    provider_provenance: tuple[tuple[str, str | bool], ...] = ()
    config_sha256: str | None = None
    skill_evolution_v2_json: str | None = None

    def __post_init__(self) -> None:
        if self.source_scope != "runtime-v2":
            raise ValueError("new alternating manifests require runtime-v2 source provenance")
        if (self.upstream_repository, self.upstream_commit, self.upstream_package_version) != (
            TAU_BENCH_REPOSITORY, TAU_BENCH_COMMIT, TAU2_PACKAGE_VERSION,
        ):
            raise ValueError("alternating runs require the pinned tau-bench release")
        if not self.experiment_id.strip() or self.domain not in ("retail", "airline"):
            raise ValueError("alternating runs require a named supported native-domain experiment")
        panels = (self.evolution_task_ids, self.validation_task_ids, self.heldout_task_ids)
        if any(not panel or len(set(panel)) != len(panel) for panel in panels):
            raise ValueError("E, V, and H must each contain unique task IDs")
        flat = tuple(task_id for panel in panels for task_id in panel)
        if any(not task_id.strip() for task_id in flat) or len(set(flat)) != len(flat):
            raise ValueError("E, V, and H task panels must be non-empty and disjoint")
        if set(flat) & set(self.excluded_task_ids):
            raise ValueError("excluded tasks cannot appear in E, V, or H")
        if self.split_name != "train" or self.heldout_split_name != "test":
            raise ValueError("E/V use τ-bench train tasks and H uses τ-bench test tasks")
        if self.seed < 0 or self.generations < 1 or self.customer_candidates < 1:
            raise ValueError("seed, generation count, and candidate count must be positive")
        if not 1 <= self.clean_panel_size <= len(self.validation_task_ids):
            raise ValueError("clean_panel_size must select tasks from the V panel")
        if self.evolution_fitness_seed is not None and (
            type(self.evolution_fitness_seed) is not int or self.evolution_fitness_seed < 0
        ):
            raise ValueError("evolution_fitness_seed must be a non-negative integer")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if (type(self.max_parallel_episodes) is not int
                or not 1 <= self.max_parallel_episodes <= MAX_PARALLEL_EPISODES):
            raise ValueError(
                f"max_parallel_episodes must be an integer from 1 to {MAX_PARALLEL_EPISODES}"
            )
        if type(self.run_validation) is not bool or type(self.run_heldout) is not bool:
            raise ValueError("run_validation and run_heldout must be booleans")
        if self.request_budget_cap is not None and self.request_budget_cap < 1:
            raise ValueError("request budget cap must be positive or null")
        if not isinstance(self.initial_customer_strategy, str) or not isinstance(
            self.initial_service_strategy, str,
        ):
            raise TypeError("initial strategies must be natural-language strings")
        if type(self.enforce_communication_protocol) is not bool:
            raise ValueError("communication protocol mode must be boolean")
        if self.customer_carrier != "prompt_strategy":
            raise ValueError("V1 supports only the existing Customer PromptStrategy carrier")
        if self.service_carrier not in {"prompt_strategy", "skill_memory_v1", "skill_memory_v2"}:
            raise ValueError("unsupported Service evolution carrier")
        if self.service_carrier != "skill_memory_v2" and (type(self.max_service_mutations_per_generation) is not int
                or self.service_skill_runtime != "inject_all"
                or self.service_mutation_ops != ("add", "update", "no_op")
                or self.max_service_mutations_per_generation != 1):
            raise ValueError("SkillMemory V1 requires inject_all and one ADD/UPDATE/NO_OP per generation")
        if self.service_carrier == 'skill_memory_v2':
            if self.skill_evolution_v2_json is None or self.initial_service_strategy.strip():
                raise ValueError('V2 requires frozen policy and empty initial runtime memory')
            import json
            policy = json.loads(self.skill_evolution_v2_json)
            if self.service_skill_runtime != policy['service_skill_runtime']:
                raise ValueError('activation runtime differs from frozen V2 policy')
        if self.service_carrier == "skill_memory_v1" and self.initial_service_strategy.strip():
            raise ValueError("SkillMemory V1 must start empty without bootstrap Service skills")
        provenance = dict(self.provider_provenance)
        if len(provenance) != len(self.provider_provenance):
            raise ValueError("provider provenance keys must be unique")
        if provenance:
            required = {
                "provider",
                "evolver_model_id",
                "backend_checkpoint",
                "reasoning_parameter_verified_by_provider",
            }
            if not required <= set(provenance):
                raise ValueError(
                    "provider provenance must record provider, Evolver model, backend checkpoint, "
                    "and provider-side reasoning verification status"
                )
            if any(not isinstance(value, (str, bool)) for value in provenance.values()):
                raise TypeError("provider provenance values must be strings or booleans")
            if type(provenance["reasoning_parameter_verified_by_provider"]) is not bool:
                raise TypeError("reasoning provider verification status must be boolean")
        models = dict(self.role_models)
        if tuple(name for name, _ in self.role_models) != EVOLUTION_ROLE_NAMES:
            raise ValueError("models must freeze agent, customer, evaluator, and evolver")
        if self.real_provider_enabled and any(not models[name] for name in EVOLUTION_ROLE_NAMES):
            raise ValueError("provider-enabled runs require a model for every τ-bench role")
        blobs = dict(self.source_blob_sha1)
        missing = alternating_source_paths(self.domain) - set(blobs)
        if missing:
            raise ValueError(f"missing pinned τ-bench source fingerprints: {sorted(missing)}")
        _relative_path(self.output_path, "output_path")
        _relative_path(self.checkpoint_path, "checkpoint_path")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AlternatingManifest:
        experiment = raw["experiment"]
        if experiment.get("phase") != "alternating-self-evolution":
            raise ValueError(
                "AlternatingManifest requires phase='alternating-self-evolution'"
            )
        upstream = experiment["upstream"]
        selection = experiment["task_selection"]
        if selection.get("source_split") != "train":
            raise ValueError("E/V tasks must use the official τ-bench train split")
        if not selection.get("heldout"):
            raise ValueError(
                "alternating runs require a sealed H panel from the τ-bench test split"
            )
        models = experiment.get("models", {})
        if not isinstance(models, Mapping) or not set(EVOLUTION_ROLE_NAMES) <= set(
            models
        ):
            raise ValueError(
                "models must include agent, customer, evaluator, and evolver"
            )
        role_models = _freeze_role_models(
            {role: models[role] for role in EVOLUTION_ROLE_NAMES},
            roles=EVOLUTION_ROLE_NAMES,
        )
        raw_model_args = experiment.get("model_args", DEFAULT_ROLE_MODEL_ARGS)
        if not isinstance(raw_model_args, Mapping) or not set(EVOLUTION_ROLE_NAMES) <= set(
            raw_model_args,
        ):
            raise ValueError("model_args must include agent, customer, evaluator, and evolver")
        role_args = freeze_role_model_args(
            {role: raw_model_args[role] for role in EVOLUTION_ROLE_NAMES},
            roles=EVOLUTION_ROLE_NAMES,
        )
        enabled = experiment.get("real_provider_enabled", False)
        if type(enabled) is not bool:
            raise ValueError("real_provider_enabled must be boolean")
        cap = experiment.get("request_budget_cap")
        evolution_fitness_seed = experiment.get(
            "evolution_fitness_seed", experiment["seed"]
        )
        if type(evolution_fitness_seed) is not int:
            raise ValueError("evolution_fitness_seed must be an integer")
        run_validation = experiment.get("run_validation", True)
        run_heldout = experiment.get("run_heldout", True)
        if type(run_validation) is not bool or type(run_heldout) is not bool:
            raise ValueError("run_validation and run_heldout must be booleans")
        evolution = experiment.get("evolution", {})
        if not isinstance(evolution, Mapping):
            raise TypeError("experiment.evolution must be a mapping")
        mutation_ops = evolution.get("service_mutation_ops", ("add", "update", "no_op"))
        if not isinstance(mutation_ops, (list, tuple)):
            raise TypeError("service_mutation_ops must be a list")
        provider_provenance = experiment.get("provider_provenance", {})
        if not isinstance(provider_provenance, Mapping):
            raise TypeError("experiment.provider_provenance must be a mapping")
        if any(not isinstance(key, str) for key in provider_provenance):
            raise TypeError("provider provenance keys must be strings")
        if any(
            not isinstance(value, (str, bool)) for value in provider_provenance.values()
        ):
            raise TypeError("provider provenance values must be strings or booleans")
        v2_policy = None
        if evolution.get('service_carrier') == 'skill_memory_v2':
            import json

            from .skill_evolution_config import freeze_v2_policy
            v2_policy = freeze_v2_policy(dict(experiment.get('skill_evolution_v2', {})), dict(role_models), _role_model_args_payload(role_args))
            v2_policy = json.dumps(v2_policy, sort_keys=True, separators=(',', ':'))
        code = capture_code_provenance(runtime_only=True)
        return cls(
            experiment_id=str(experiment["id"]),
            upstream_repository=str(upstream["repository"]),
            upstream_commit=str(upstream["commit"]),
            upstream_package_version=str(upstream["package_version"]),
            evotau_git_commit=code.git_commit,
            evotau_working_tree_clean=code.working_tree_clean,
            evotau_source_sha256=code.source_sha256,
            evolution_task_ids=tuple(
                str(item) for item in selection.get("evolution", ())
            ),
            validation_task_ids=tuple(
                str(item) for item in selection.get("validation", ())
            ),
            heldout_task_ids=tuple(str(item) for item in selection.get("heldout", ())),
            excluded_task_ids=tuple(
                str(item) for item in selection.get("excluded", ())
            ),
            seed=int(experiment["seed"]),
            generations=int(experiment["generations"]),
            customer_candidates=int(experiment.get("customer_candidates", 2)),
            clean_panel_size=int(experiment.get("clean_panel_size", 1)),
            max_steps=int(experiment.get("max_steps", 64)),
            max_parallel_episodes=experiment.get(
                "max_parallel_episodes",
                DEFAULT_MAX_PARALLEL_EPISODES,
            ),
            request_budget_cap=None if cap is None else int(cap),
            real_provider_enabled=enabled,
            role_models=role_models,
            role_model_args=role_args,
            source_blob_sha1=tuple(
                sorted(
                    (str(path), str(digest).lower())
                    for path, digest in experiment["source_blob_sha1"].items()
                    if str(path) in alternating_source_paths(str(experiment.get("domain", "retail")))
                )
            ),
            initial_customer_strategy=str(experiment.get("customer_strategy") or ""),
            initial_service_strategy=str(experiment.get("service_strategy") or ""),
            output_path=_relative_path(str(experiment["output_path"]), "output_path"),
            checkpoint_path=_relative_path(
                str(experiment["checkpoint_path"]), "checkpoint_path"
            ),
            evolution_fitness_seed=evolution_fitness_seed,
            run_validation=run_validation,
            run_heldout=run_heldout,
            domain=str(experiment.get("domain", "retail")),
            split_name=str(selection["source_split"]),
            heldout_split_name=str(selection.get("heldout_split", "test")),
            enforce_communication_protocol=experiment.get(
                "enforce_communication_protocol",
                False,
            ),
            customer_carrier=str(evolution.get("customer_carrier", "prompt_strategy")),
            service_carrier=str(evolution.get("service_carrier", "prompt_strategy")),
            service_skill_runtime=str(
                evolution.get(
                    "service_skill_runtime",
                    json.loads(v2_policy)["service_skill_runtime"]
                    if v2_policy
                    else "inject_all",
                )
            ),
            service_mutation_ops=tuple(str(item) for item in mutation_ops),
            max_service_mutations_per_generation=evolution.get(
                "max_service_mutations_per_generation",
                1,
            ),
            provider_provenance=tuple(sorted(provider_provenance.items())),
            config_sha256=sha256_json(raw),
            skill_evolution_v2_json=v2_policy,
        )

    def bind_saved_provenance(self, saved: Mapping[str, Any]) -> AlternatingManifest:
        """Keep the original identity only when every actual run input still matches."""
        payload = {key: value for key, value in saved.items() if key != "manifest_sha256"}
        if saved.get("manifest_sha256") != sha256_json(payload):
            raise ValueError("existing frozen manifest has an invalid fingerprint")
        provenance = saved.get("evotau")
        if not isinstance(provenance, Mapping):
            raise TypeError("existing frozen manifest is missing code provenance")
        bound = replace(
            self,
            evotau_git_commit=provenance.get("git_commit"),
            evotau_working_tree_clean=provenance.get("working_tree_clean"),
        )
        if bound.to_document() != saved:
            raise ValueError("existing run directory belongs to different frozen execution inputs")
        return bound

    @property
    def role_model_args_dict(self) -> dict[str, dict[str, float | int | str]]:
        return _role_model_args_payload(self.role_model_args)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "experiment_id": self.experiment_id,
            "phase": "alternating-self-evolution",
            "upstream": {
                "repository": self.upstream_repository,
                "commit": self.upstream_commit,
                "package_version": self.upstream_package_version,
            },
            "evotau": {
                "git_commit": self.evotau_git_commit,
                "working_tree_clean": self.evotau_working_tree_clean,
                "source_sha256": self.evotau_source_sha256,
                "source_scope": self.source_scope,
                "runtime_source_sha256": self.evotau_source_sha256,
            },
            "domain": self.domain,
            "task_panels": {
                "E": list(self.evolution_task_ids),
                "V": list(self.validation_task_ids),
                "H": list(self.heldout_task_ids),
                "excluded": list(self.excluded_task_ids),
                "split": self.split_name,
                "heldout_split": self.heldout_split_name,
            },
            "seed": self.seed,
            "evolution_fitness_seed": (
                self.seed if self.evolution_fitness_seed is None else self.evolution_fitness_seed
            ),
            "generations": self.generations,
            "customer_candidates": self.customer_candidates,
            "clean_panel_size": self.clean_panel_size,
            "max_steps": self.max_steps,
            "max_parallel_episodes": self.max_parallel_episodes,
            "run_validation": self.run_validation,
            "run_heldout": self.run_heldout,
            "request_budget_cap": self.request_budget_cap,
            "real_provider_enabled": self.real_provider_enabled,
            "role_models": dict(self.role_models),
            "role_model_args": self.role_model_args_dict,
            "source_blob_sha1": dict(self.source_blob_sha1),
            "initial_customer_strategy_sha256": sha256_json(self.initial_customer_strategy),
            "initial_service_strategy_sha256": sha256_json(self.initial_service_strategy),
            "communication_enforcement": self.enforce_communication_protocol,
            "output_path": self.output_path,
            "checkpoint_path": self.checkpoint_path,
        }
        if self.service_carrier != "prompt_strategy":
            payload["evolution"] = {
                "customer_carrier": self.customer_carrier,
                "service_carrier": self.service_carrier,
                "service_skill_runtime": self.service_skill_runtime,
                "service_mutation_ops": list(self.service_mutation_ops),
                "max_service_mutations_per_generation": self.max_service_mutations_per_generation,
            }
        if self.skill_evolution_v2_json is not None:
            import json
            payload['skill_evolution_v2'] = json.loads(self.skill_evolution_v2_json)
        if self.provider_provenance:
            payload["provider_provenance"] = dict(self.provider_provenance)
        if self.config_sha256 is not None:
            payload["config_sha256"] = self.config_sha256
        return payload

    def to_document(self) -> dict[str, Any]:
        payload = self.to_payload()
        payload["manifest_sha256"] = self.sha256
        return payload
