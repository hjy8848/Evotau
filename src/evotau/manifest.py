"""Single-episode native τ-bench integration-check manifest.

The evolutionary method uses :mod:`evotau.alternating_manifest`; this small
manifest remains only for the optional native-runtime connectivity check.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .tau_provenance import (
    DEFAULT_ROLE_MODEL_ARGS,
    REQUIRED_SOURCE_PATHS,
    ROLE_NAMES,
    TAU2_PACKAGE_VERSION,
    TAU_BENCH_COMMIT,
    TAU_BENCH_REPOSITORY,
    _freeze_role_models,
    _relative_path,
    _role_model_args_payload,
    capture_code_provenance,
    freeze_role_model_args,
    sha256_json,
)


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    experiment_id: str
    phase: str
    upstream_repository: str
    upstream_commit: str
    upstream_package_version: str
    evotau_git_commit: str | None
    evotau_working_tree_clean: bool | None
    evotau_source_sha256: str
    domain: str
    communication_mode: str
    enforce_communication_protocol: bool
    evaluation_type: str
    split_name: str
    evolution_task_ids: tuple[str, ...]
    validation_task_ids: tuple[str, ...]
    excluded_task_ids: tuple[str, ...]
    seed: int
    max_steps: int
    request_budget_cap: int
    provider_retries: int
    real_provider_enabled: bool
    role_models: tuple[tuple[str, str | None], ...]
    role_model_args: tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...]
    source_blob_sha1: tuple[tuple[str, str], ...]
    customer_strategy_sha256: str
    service_strategy_sha256: str
    output_path: str
    checkpoint_path: str

    def __post_init__(self) -> None:
        if self.phase != "0-integration-proof":
            raise ValueError("ExperimentManifest is only for the one-episode integration proof")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if (self.upstream_repository, self.upstream_commit, self.upstream_package_version) != (
            TAU_BENCH_REPOSITORY, TAU_BENCH_COMMIT, TAU2_PACKAGE_VERSION,
        ):
            raise ValueError("integration proof requires the pinned τ-bench release")
        if self.domain != "retail" or self.communication_mode != "half_duplex_text":
            raise ValueError("integration proof is fixed to τ-bench Retail text mode")
        if type(self.enforce_communication_protocol) is not bool:
            raise ValueError("enforce_communication_protocol must be boolean")
        if self.evaluation_type != "all" or self.split_name != "train":
            raise ValueError("integration proof uses native ALL evaluation and official train tasks")
        if len(self.evolution_task_ids) != 1 or len(self.validation_task_ids) != 1:
            raise ValueError("integration proof freezes one E and one V task")
        selected = self.evolution_task_ids + self.validation_task_ids
        if len(set(selected)) != len(selected) or set(selected) & set(self.excluded_task_ids):
            raise ValueError("E, V, and excluded task IDs must be disjoint")
        if self.seed < 0 or self.max_steps < 1 or self.request_budget_cap < 1:
            raise ValueError("seed, max_steps, and request budget must be positive")
        if self.provider_retries != 0:
            raise ValueError("provider retries must be disabled")
        if type(self.real_provider_enabled) is not bool:
            raise ValueError("real_provider_enabled must be boolean")
        if self.real_provider_enabled and any(not model for _, model in self.role_models):
            raise ValueError("live integration checks require frozen model IDs")
        if tuple(name for name, _ in self.role_models) != ROLE_NAMES:
            raise ValueError(f"role_models must contain exactly {sorted(ROLE_NAMES)}")
        freeze_role_model_args(_role_model_args_payload(self.role_model_args), roles=ROLE_NAMES)
        for digest in (self.customer_strategy_sha256, self.service_strategy_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("strategy fingerprints must be SHA-256 hex digests")
        blobs = dict(self.source_blob_sha1)
        if missing := REQUIRED_SOURCE_PATHS - set(blobs):
            raise ValueError(f"missing upstream source fingerprints: {sorted(missing)}")
        if any(not re.fullmatch(r"[0-9a-f]{40}", digest) for digest in blobs.values()):
            raise ValueError("upstream source fingerprints must be Git blob SHA-1 values")
        _relative_path(self.output_path, "output_path")
        _relative_path(self.checkpoint_path, "checkpoint_path")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExperimentManifest:
        experiment = raw["experiment"]
        selection = experiment["task_selection"]
        customer = experiment.get("customer_strategy")
        service = experiment.get("service_strategy")
        code = capture_code_provenance()
        enabled = experiment.get("real_provider_enabled", False)
        return cls(
            experiment_id=str(experiment["id"]),
            phase=str(experiment["phase"]),
            upstream_repository=str(experiment["upstream"]["repository"]),
            upstream_commit=str(experiment["upstream"]["commit"]),
            upstream_package_version=str(experiment["upstream"]["package_version"]),
            evotau_git_commit=code.git_commit,
            evotau_working_tree_clean=code.working_tree_clean,
            evotau_source_sha256=code.source_sha256,
            domain=str(experiment["domain"]),
            communication_mode=str(experiment["communication_mode"]),
            enforce_communication_protocol=experiment.get("enforce_communication_protocol", False),
            evaluation_type=str(experiment["evaluation_type"]),
            split_name=str(selection["source_split"]),
            evolution_task_ids=tuple(str(item) for item in selection["evolution"]),
            validation_task_ids=tuple(str(item) for item in selection["validation"]),
            excluded_task_ids=tuple(str(item) for item in selection.get("excluded", ())),
            seed=int(experiment["seed"]),
            max_steps=int(experiment["max_steps"]),
            request_budget_cap=int(experiment["request_budget_cap"]),
            provider_retries=int(experiment["provider_retries"]),
            real_provider_enabled=enabled,
            role_models=_freeze_role_models(experiment.get("models", {}), roles=ROLE_NAMES),
            role_model_args=freeze_role_model_args(
                experiment.get(
                    "model_args",
                    {role: DEFAULT_ROLE_MODEL_ARGS[role] for role in ROLE_NAMES},
                ),
                roles=ROLE_NAMES,
            ),
            source_blob_sha1=tuple(sorted(
                (str(path), str(digest).lower())
                for path, digest in experiment["source_blob_sha1"].items()
            )),
            customer_strategy_sha256=sha256_json(
                {"mode": "native_no_overlay"} if customer is None else customer
            ),
            service_strategy_sha256=sha256_json(
                {"text": ""} if service is None or service == {"rules": []} else service
            ),
            output_path=_relative_path(str(experiment["output_path"]), "output_path"),
            checkpoint_path=_relative_path(str(experiment["checkpoint_path"]), "checkpoint_path"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "upstream": {
                "repository": self.upstream_repository,
                "commit": self.upstream_commit,
                "package_version": self.upstream_package_version,
            },
            "evotau": {
                "git_commit": self.evotau_git_commit,
                "working_tree_clean": self.evotau_working_tree_clean,
                "source_sha256": self.evotau_source_sha256,
            },
            "domain": self.domain,
            "communication_mode": self.communication_mode,
            "enforce_communication_protocol": self.enforce_communication_protocol,
            "evaluation_type": self.evaluation_type,
            "task_selection": {
                "source_split": self.split_name,
                "evolution": list(self.evolution_task_ids),
                "validation": list(self.validation_task_ids),
                "excluded": list(self.excluded_task_ids),
            },
            "seed": self.seed,
            "max_steps": self.max_steps,
            "request_budget_cap": self.request_budget_cap,
            "provider_retries": self.provider_retries,
            "real_provider_enabled": self.real_provider_enabled,
            "role_models": dict(self.role_models),
            "role_model_args": _role_model_args_payload(self.role_model_args),
            "source_blob_sha1": dict(self.source_blob_sha1),
            "strategy_sha256": {
                "customer": self.customer_strategy_sha256,
                "service": self.service_strategy_sha256,
            },
            "paths": {"output": self.output_path, "checkpoint": self.checkpoint_path},
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())

    def to_document(self) -> dict[str, Any]:
        return {**self.to_payload(), "manifest_sha256": self.sha256}
