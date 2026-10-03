"""Immutable Phase 0 manifest construction and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

TAU_BENCH_REPOSITORY = "sierra-research/tau2-bench"
TAU_BENCH_COMMIT = "b7ea9074c1cba482b30687fecdb5c8425fd6f619"
TAU2_PACKAGE_VERSION = "1.0.1"
ROLE_NAMES = ("agent", "customer", "reviewer", "evaluator")
MECHANISM_ROLE_NAMES = (*ROLE_NAMES, "evolver")
MODEL_ARGUMENT_NAMES = frozenset({
    "temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty",
    "api_base", "thinking_mode",
})
DEFAULT_ROLE_MODEL_ARGS = {role: {"temperature": 0.0} for role in MECHANISM_ROLE_NAMES}
MVP_FAILURE_TAXONOMY = (
    ("identity_verification", "retail.policy:identity_verification", "missing_identity_verification"),
    ("pre_write", "retail.policy:explicit_confirmation", "missing_explicit_confirmation"),
    ("pre_write", "retail.policy:complete_change_scope", "incomplete_write_scope"),
)

REQUIRED_SOURCE_PATHS = frozenset(
    {
        "data/tau2/domains/retail/tasks.json",
        "data/tau2/domains/retail/split_tasks.json",
        "data/tau2/domains/retail/db.json",
        "data/tau2/domains/retail/policy.md",
        "data/tau2/user_simulator/simulation_guidelines.md",
        "src/tau2/domains/retail/tools.py",
        "src/tau2/environment/environment.py",
        "src/tau2/registry.py",
        "src/tau2/orchestrator/orchestrator.py",
        "src/tau2/user/user_simulator.py",
        "src/tau2/agent/llm_agent.py",
        "src/tau2/runner/build.py",
        "src/tau2/runner/simulation.py",
        "src/tau2/data_model/simulation.py",
        "src/tau2/evaluator/evaluator.py",
        "src/tau2/evaluator/evaluator_nl_assertions.py",
        "src/tau2/evaluator/review_llm_judge.py",
        "src/tau2/evaluator/reviewer.py",
        "src/tau2/utils/llm_utils.py",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _communication_protocol_enforcement(experiment: Mapping[str, Any]) -> bool:
    value = experiment.get("enforce_communication_protocol", False)
    if type(value) is not bool:
        raise ValueError("enforce_communication_protocol must be a boolean")
    return value


DEFAULT_CUSTOMER_STRATEGY_HASH = sha256_json({
    "disclosure": "minimal_on_request", "request_order": "scenario_order",
    "challenge_style": "none", "challenge_budget": 0,
})
DEFAULT_SERVICE_STRATEGY_HASH = sha256_json({"rules": []})


@dataclass(frozen=True, slots=True)
class CodeProvenance:
    git_commit: str | None
    working_tree_clean: bool | None
    source_sha256: str

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "git_commit": self.git_commit,
            "working_tree_clean": self.working_tree_clean,
            "source_sha256": self.source_sha256,
        }


def capture_code_provenance() -> CodeProvenance:
    """Fingerprint EvoTau sources and, when available, the containing Git state."""

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parents[1]
    in_project = (project_root / "pyproject.toml").is_file()
    if in_project:
        source_paths = sorted(
            [*package_dir.rglob("*.py"), project_root / "pyproject.toml", *project_root.glob("configs/**/*.yaml")]
        )
        relative = lambda path: path.relative_to(project_root).as_posix()
    else:
        source_paths = sorted(package_dir.rglob("*.py"))
        relative = lambda path: f"evotau/{path.relative_to(package_dir).as_posix()}"
    digest = hashlib.sha256()
    for path in source_paths:
        if not path.is_file():
            continue
        name = relative(path).encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    commit = None
    clean = None
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            check=False, capture_output=True, text=True, timeout=2,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root,
            check=False, capture_output=True, text=True, timeout=2,
        )
        if commit_result.returncode == 0 and status_result.returncode == 0:
            commit = commit_result.stdout.strip()
            clean = not status_result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return CodeProvenance(commit, clean, digest.hexdigest())


def _validate_code_provenance(git_commit: str | None, source_sha256: str) -> None:
    if git_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValueError("EvoTau Git commit must be a full lowercase SHA-1")
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("EvoTau source fingerprint must be a SHA-256 hex digest")


def git_blob_sha1(data: bytes) -> str:
    """Return Git's SHA-1 for a blob, including its object header."""

    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def verify_git_blob_sha1(path: str | Path, expected_sha1: str) -> None:
    actual = git_blob_sha1(Path(path).read_bytes())
    if actual != expected_sha1:
        raise ValueError(
            f"upstream file fingerprint mismatch for {path}: "
            f"expected {expected_sha1}, got {actual}"
        )


def _relative_path(value: str, field: str) -> str:
    if not value or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{field} must be a non-empty relative path")
    if ".." in PurePosixPath(value).parts or ".." in PureWindowsPath(value).parts:
        raise ValueError(f"{field} cannot escape the project directory")
    return value.replace("\\", "/")


def freeze_role_model_args(
    raw: Mapping[str, Any] | None,
    *,
    roles: tuple[str, ...] = ROLE_NAMES,
) -> tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...]:
    """Validate and freeze the generation arguments applied to each model role."""

    if raw is None:
        raw = {role: DEFAULT_ROLE_MODEL_ARGS[role] for role in roles}
    if not isinstance(raw, Mapping) or set(raw) != set(roles):
        raise ValueError(f"model_args must contain exactly {sorted(roles)}")
    frozen = []
    for role in roles:
        params = raw[role]
        if not isinstance(params, Mapping) or not params:
            raise ValueError(f"model_args.{role} must be a non-empty mapping")
        if set(params) - MODEL_ARGUMENT_NAMES:
            raise ValueError(
                f"model_args.{role} contains unsupported arguments: "
                f"{sorted(set(params) - MODEL_ARGUMENT_NAMES)}"
            )
        normalized: list[tuple[str, float | int | str]] = []
        for name, value in sorted(params.items()):
            if name == "max_tokens":
                if type(value) is not int or value <= 0:
                    raise ValueError(f"model_args.{role}.max_tokens must be a positive integer")
                normalized.append((name, value))
                continue
            if name == "thinking_mode":
                if not isinstance(value, str) or value not in {"disabled", "enabled"}:
                    raise ValueError(
                        f"model_args.{role}.thinking_mode must be 'disabled' or 'enabled'"
                    )
                normalized.append((name, value))
                continue
            if name == "api_base":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"model_args.{role}.api_base must be an HTTP(S) URL")
                parsed = urlsplit(value)
                if (
                    parsed.scheme not in {"http", "https"}
                    or not parsed.netloc
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError(
                        f"model_args.{role}.api_base must be an HTTP(S) URL without credentials"
                    )
                normalized.append((name, value.rstrip("/")))
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"model_args.{role}.{name} must be numeric")
            low, high = (0.0, 2.0) if name == "temperature" else (
                (0.0, 1.0) if name == "top_p" else (-2.0, 2.0)
            )
            if not low <= value <= high:
                raise ValueError(f"model_args.{role}.{name} must be in {low}..{high}")
            normalized.append((name, value))
        frozen.append((role, tuple(normalized)))
    return tuple(frozen)


def _freeze_role_models(
    raw: Mapping[str, Any],
    *,
    roles: tuple[str, ...] = ROLE_NAMES,
) -> tuple[tuple[str, str | None], ...]:
    if not isinstance(raw, Mapping) or set(raw) != set(roles):
        raise ValueError(f"models must contain exactly {sorted(roles)}")
    frozen = []
    for role in roles:
        value = raw[role]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"models.{role} must be a non-empty model ID or null")
        frozen.append((role, value))
    return tuple(frozen)


def _role_model_args_payload(
    values: tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...],
) -> dict[str, dict[str, float | int | str]]:
    return {role: dict(params) for role, params in values}


def role_model_args_for_runtime(
    values: tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...],
) -> dict[str, dict[str, Any]]:
    """Translate frozen role settings into LiteLLM completion keyword arguments."""

    params = _role_model_args_payload(values)
    for role in params:
        thinking_mode = params[role].pop("thinking_mode", None)
        if thinking_mode is not None:
            params[role]["extra_body"] = {"thinking": {"type": thinking_mode}}
    return params


def _role_runtime_arguments(
    values: tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...],
) -> dict[str, Any]:
    params = role_model_args_for_runtime(values)
    result = {
        "agent_llm_args": {**params["agent"], "num_retries": 0},
        "customer_llm_args": {**params["customer"], "num_retries": 0},
        "reviewer_llm_args": params["reviewer"],
        "evaluator_llm_args": params["evaluator"],
        "provider_default_max_retries": 0,
        "cache_enabled": False,
        "review_mode": "full",
    }
    if "evolver" in params:
        result["evolver_llm_args"] = params["evolver"]
    return result


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
    heldout_task_ids: tuple[str, ...]
    excluded_task_ids: tuple[str, ...]
    seed: int
    max_steps: int
    max_episodes: int
    request_budget_cap: int
    provider_retries: int
    max_concurrency: int
    real_provider_enabled: bool
    role_models: tuple[tuple[str, str | None], ...]
    role_model_args: tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...]
    source_blob_sha1: tuple[tuple[str, str], ...]
    customer_strategy_sha256: str
    service_strategy_sha256: str
    output_path: str
    checkpoint_path: str

    def __post_init__(self) -> None:
        _validate_code_provenance(self.evotau_git_commit, self.evotau_source_sha256)
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if self.upstream_repository != TAU_BENCH_REPOSITORY:
            raise ValueError("upstream repository must match the pinned tau-bench repository")
        if self.upstream_commit != TAU_BENCH_COMMIT:
            raise ValueError("upstream commit must match the audited tau-bench commit")
        if self.upstream_package_version != TAU2_PACKAGE_VERSION:
            raise ValueError("upstream package version does not match the pinned commit")
        if self.domain != "retail" or self.communication_mode != "half_duplex_text":
            raise ValueError("Phase 0 is fixed to tau-bench Retail half-duplex text mode")
        if type(self.enforce_communication_protocol) is not bool:
            raise ValueError("enforce_communication_protocol must be a boolean")
        if self.evaluation_type != "all":
            raise ValueError("Phase 0 must explicitly use EvaluationType.ALL")
        if self.split_name != "train":
            raise ValueError("smoke E and V tasks must come from the official train split")
        if len(self.evolution_task_ids) != 1 or len(self.validation_task_ids) != 1:
            raise ValueError("Phase 0 smoke requires exactly one E task and one V task")
        selected = (
            self.evolution_task_ids + self.validation_task_ids + self.heldout_task_ids
        )
        if len(selected) != len(set(selected)):
            raise ValueError("E, V, and H task IDs must be disjoint")
        if self.heldout_task_ids:
            raise ValueError("the minimal smoke uses H=0; final heldout tasks stay sealed")
        if set(selected) & set(self.excluded_task_ids):
            raise ValueError("an excluded task cannot be selected for E, V, or H")
        if self.max_episodes != 1:
            raise ValueError("Phase 0 permits only one real episode")
        if self.max_steps != 64:
            raise ValueError("Phase 0 max_steps must be 64")
        if not 1 <= self.request_budget_cap <= 70:
            raise ValueError("Phase 0 request budget must be in the range 1..70")
        if self.provider_retries != 0:
            raise ValueError("provider retries must be disabled for Phase 0")
        if self.max_concurrency != 1:
            raise ValueError("Phase 0 runs with concurrency=1")
        if self.real_provider_enabled:
            missing = [name for name, model in self.role_models if not model]
            if missing:
                raise ValueError(f"live runs require frozen model IDs for: {', '.join(missing)}")
        if tuple(name for name, _ in self.role_models) != ROLE_NAMES:
            raise ValueError(f"role_models must freeze exactly {sorted(ROLE_NAMES)}")
        freeze_role_model_args(_role_model_args_payload(self.role_model_args))
        if not re.fullmatch(r"[0-9a-f]{64}", self.customer_strategy_sha256):
            raise ValueError("customer strategy hash must be a SHA-256 hex digest")
        if not re.fullmatch(r"[0-9a-f]{64}", self.service_strategy_sha256):
            raise ValueError("service strategy hash must be a SHA-256 hex digest")
        blobs = dict(self.source_blob_sha1)
        missing_sources = REQUIRED_SOURCE_PATHS - set(blobs)
        if missing_sources:
            raise ValueError(f"missing upstream source fingerprints: {sorted(missing_sources)}")
        invalid = [
            path for path, sha in blobs.items() if not re.fullmatch(r"[0-9a-f]{40}", sha)
        ]
        if invalid:
            raise ValueError(f"invalid Git blob SHA-1 values for: {sorted(invalid)}")
        _relative_path(self.output_path, "output_path")
        _relative_path(self.checkpoint_path, "checkpoint_path")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExperimentManifest:
        experiment = raw["experiment"]
        selection = experiment["task_selection"]
        models = experiment.get("models", {})
        source_blobs = experiment["source_blob_sha1"]
        customer = experiment.get("customer_strategy")
        service = experiment.get("service_strategy")
        code = capture_code_provenance()
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
            enforce_communication_protocol=_communication_protocol_enforcement(experiment),
            evaluation_type=str(experiment["evaluation_type"]),
            split_name=str(selection["source_split"]),
            evolution_task_ids=tuple(str(x) for x in selection["evolution"]),
            validation_task_ids=tuple(str(x) for x in selection["validation"]),
            heldout_task_ids=tuple(str(x) for x in selection.get("heldout", ())),
            excluded_task_ids=tuple(str(x) for x in selection.get("excluded", ())),
            seed=int(experiment["seed"]),
            max_steps=int(experiment["max_steps"]),
            max_episodes=int(experiment["max_episodes"]),
            request_budget_cap=int(experiment["request_budget_cap"]),
            provider_retries=int(experiment["provider_retries"]),
            max_concurrency=int(experiment["max_concurrency"]),
            real_provider_enabled=bool(experiment["real_provider_enabled"]),
            role_models=_freeze_role_models(models),
            role_model_args=freeze_role_model_args(experiment.get("model_args")),
            source_blob_sha1=tuple(
                sorted((str(path), str(sha).lower()) for path, sha in source_blobs.items())
            ),
            customer_strategy_sha256=sha256_json(
                {"mode": "native_no_overlay"} if customer is None else customer
            ),
            service_strategy_sha256=sha256_json(
                {"rules": []} if service is None else service
            ),
            output_path=_relative_path(
                str(experiment["output_path"]), "output_path"
            ),
            checkpoint_path=_relative_path(
                str(experiment["checkpoint_path"]), "checkpoint_path"
            ),
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
                "heldout": list(self.heldout_task_ids),
                "excluded": list(self.excluded_task_ids),
            },
            "seed": self.seed,
            "max_steps": self.max_steps,
            "max_episodes": self.max_episodes,
            "request_budget_cap": self.request_budget_cap,
            "provider_retries": self.provider_retries,
            "max_concurrency": self.max_concurrency,
            "real_provider_enabled": self.real_provider_enabled,
            "role_models": dict(self.role_models),
            "role_model_args": _role_model_args_payload(self.role_model_args),
            "runtime_arguments": _role_runtime_arguments(self.role_model_args),
            "source_blob_sha1": dict(self.source_blob_sha1),
            "strategy_sha256": {
                "customer": self.customer_strategy_sha256,
                "service": self.service_strategy_sha256,
            },
            "paths": {
                "output": self.output_path,
                "checkpoint": self.checkpoint_path,
            },
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())

    def to_document(self) -> dict[str, Any]:
        payload = self.to_payload()
        payload["manifest_sha256"] = self.sha256
        return payload


@dataclass(frozen=True, slots=True)
class MechanismManifest:
    """Frozen protocol for the bounded two-generation Phase 3 smoke."""

    experiment_id: str
    upstream_repository: str
    upstream_commit: str
    upstream_package_version: str
    evotau_git_commit: str | None
    evotau_working_tree_clean: bool | None
    evotau_source_sha256: str
    evolution_task_id: str
    validation_task_id: str
    seed: int
    max_steps: int
    max_episodes: int
    request_budget_cap: int
    provider_retries: int
    max_concurrency: int
    real_provider_enabled: bool
    role_models: tuple[tuple[str, str | None], ...]
    role_model_args: tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...]
    source_blob_sha1: tuple[tuple[str, str], ...]
    output_path: str
    checkpoint_path: str
    customer_candidates: int = 2
    generations: int = 2
    domain: str = "retail"
    communication_mode: str = "half_duplex_text"
    enforce_communication_protocol: bool = False
    evaluation_type: str = "all"
    split_name: str = "train"
    excluded_task_ids: tuple[str, ...] = ()
    customer_strategy_sha256: str = DEFAULT_CUSTOMER_STRATEGY_HASH
    service_strategy_sha256: str = DEFAULT_SERVICE_STRATEGY_HASH

    def __post_init__(self) -> None:
        _validate_code_provenance(self.evotau_git_commit, self.evotau_source_sha256)
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if (self.upstream_repository, self.upstream_commit, self.upstream_package_version) != (
            TAU_BENCH_REPOSITORY, TAU_BENCH_COMMIT, TAU2_PACKAGE_VERSION
        ):
            raise ValueError("mechanism smoke must use the audited tau-bench pin")
        if self.domain != "retail" or self.communication_mode != "half_duplex_text" or self.evaluation_type != "all":
            raise ValueError("mechanism smoke is fixed to Retail half-duplex text with evaluation_type=all")
        if type(self.enforce_communication_protocol) is not bool:
            raise ValueError("enforce_communication_protocol must be a boolean")
        if self.split_name != "train":
            raise ValueError("E and V tasks must come from the official train split")
        if not self.evolution_task_id or not self.validation_task_id or self.evolution_task_id == self.validation_task_id:
            raise ValueError("E and V task IDs must be distinct and non-empty")
        if {self.evolution_task_id, self.validation_task_id} & set(self.excluded_task_ids):
            raise ValueError("excluded tasks cannot be selected for E or V")
        if self.seed < 0 or self.max_steps != 64:
            raise ValueError("mechanism smoke requires a non-negative seed and max_steps=64")
        if self.customer_candidates != 2 or self.generations != 2:
            raise ValueError("minimal mechanism smoke freezes K=2 and exactly two generations")
        if not 1 <= self.max_episodes <= 23:
            raise ValueError("Phase 3 permits at most 23 episodes including Phase 0 integration")
        if not 1 <= self.request_budget_cap <= 1800:
            raise ValueError("Phase 3 provider-attempt cap must be in the range 1..1800")
        if self.provider_retries != 0 or type(self.max_concurrency) is not int or not 1 <= self.max_concurrency <= 4:
            raise ValueError("mechanism smoke requires provider retries=0 and concurrency in the range 1..4")
        models = dict(self.role_models)
        if set(models) != set(MECHANISM_ROLE_NAMES):
            raise ValueError(f"role_models must freeze exactly {sorted(MECHANISM_ROLE_NAMES)}")
        freeze_role_model_args(
            _role_model_args_payload(self.role_model_args), roles=MECHANISM_ROLE_NAMES,
        )
        if self.real_provider_enabled and any(not models[name] for name in models):
            raise ValueError("live smoke requires frozen model IDs for every role")
        blobs = dict(self.source_blob_sha1)
        missing = REQUIRED_SOURCE_PATHS - set(blobs)
        if missing:
            raise ValueError(f"missing upstream source fingerprints: {sorted(missing)}")
        invalid = [path for path, digest in blobs.items() if not re.fullmatch(r"[0-9a-f]{40}", digest)]
        if invalid:
            raise ValueError(f"invalid Git blob SHA-1 values for: {sorted(invalid)}")
        for name, digest in (("customer", self.customer_strategy_sha256), ("service", self.service_strategy_sha256)):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{name} strategy hash must be a SHA-256 hex digest")
        _relative_path(self.output_path, "output_path")
        _relative_path(self.checkpoint_path, "checkpoint_path")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MechanismManifest:
        experiment = raw["experiment"]
        selection = experiment["task_selection"]
        models = experiment.get("models", {})
        upstream = experiment["upstream"]
        if experiment.get("phase") != "3-two-generation-smoke":
            raise ValueError("MechanismManifest requires phase='3-two-generation-smoke'")
        if len(selection.get("evolution", ())) != 1 or len(selection.get("validation", ())) != 1:
            raise ValueError("mechanism smoke requires exactly one E task and one V task")
        if selection.get("heldout", ()):
            raise ValueError("Phase 3 smoke keeps H=0; heldout tasks remain sealed")
        code = capture_code_provenance()
        return cls(
            experiment_id=str(experiment["id"]),
            upstream_repository=str(upstream["repository"]),
            upstream_commit=str(upstream["commit"]),
            upstream_package_version=str(upstream["package_version"]),
            evotau_git_commit=code.git_commit,
            evotau_working_tree_clean=code.working_tree_clean,
            evotau_source_sha256=code.source_sha256,
            evolution_task_id=str(selection["evolution"][0]),
            validation_task_id=str(selection["validation"][0]),
            seed=int(experiment["seed"]), max_steps=int(experiment["max_steps"]),
            max_episodes=int(experiment["max_episodes"]),
            request_budget_cap=int(experiment["request_budget_cap"]),
            provider_retries=int(experiment["provider_retries"]),
            max_concurrency=int(experiment["max_concurrency"]),
            real_provider_enabled=bool(experiment["real_provider_enabled"]),
            role_models=_freeze_role_models(models, roles=MECHANISM_ROLE_NAMES),
            role_model_args=freeze_role_model_args(
                experiment.get("model_args"), roles=MECHANISM_ROLE_NAMES,
            ),
            source_blob_sha1=tuple(sorted((str(path), str(digest).lower())
                                          for path, digest in experiment["source_blob_sha1"].items())),
            output_path=_relative_path(str(experiment["output_path"]), "output_path"),
            checkpoint_path=_relative_path(str(experiment["checkpoint_path"]), "checkpoint_path"),
            customer_candidates=int(experiment.get("customer_candidates", 2)),
            generations=int(experiment.get("generations", 2)),
            domain=str(experiment["domain"]),
            communication_mode=str(experiment["communication_mode"]),
            enforce_communication_protocol=_communication_protocol_enforcement(experiment),
            evaluation_type=str(experiment["evaluation_type"]),
            split_name=str(selection["source_split"]),
            excluded_task_ids=tuple(str(task_id) for task_id in selection.get("excluded", ())),
            customer_strategy_sha256=sha256_json(
                {"disclosure": "minimal_on_request", "request_order": "scenario_order",
                 "challenge_style": "none", "challenge_budget": 0}
                if experiment.get("customer_strategy") is None else experiment["customer_strategy"]
            ),
            service_strategy_sha256=sha256_json(
                {"rules": []} if experiment.get("service_strategy") is None else experiment["service_strategy"]
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "phase": "3-two-generation-smoke",
            "upstream": {"repository": self.upstream_repository, "commit": self.upstream_commit,
                         "package_version": self.upstream_package_version},
            "evotau": {"git_commit": self.evotau_git_commit,
                       "working_tree_clean": self.evotau_working_tree_clean,
                       "source_sha256": self.evotau_source_sha256},
            "domain": self.domain, "communication_mode": self.communication_mode,
            "enforce_communication_protocol": self.enforce_communication_protocol,
            "evaluation_type": self.evaluation_type, "split_name": self.split_name,
            "evolution_task_id": self.evolution_task_id, "validation_task_id": self.validation_task_id,
            "excluded_task_ids": list(self.excluded_task_ids),
            "seed": self.seed, "max_steps": self.max_steps, "max_episodes": self.max_episodes,
            "request_budget_cap": self.request_budget_cap, "provider_retries": self.provider_retries,
            "max_concurrency": self.max_concurrency, "customer_candidates": self.customer_candidates,
            "generations": self.generations, "real_provider_enabled": self.real_provider_enabled,
            "role_models": dict(self.role_models),
            "role_model_args": _role_model_args_payload(self.role_model_args),
            "source_blob_sha1": dict(self.source_blob_sha1),
            "runtime_arguments": _role_runtime_arguments(self.role_model_args),
            "failure_taxonomy": [
                {
                    "workflow_stage": stage,
                    "policy_rule_id": policy_rule_id,
                    "mistake_type": mistake,
                }
                for stage, policy_rule_id, mistake in MVP_FAILURE_TAXONOMY
            ],
            "failure_taxonomy_sha256": sha256_json(MVP_FAILURE_TAXONOMY),
            "strategy_sha256": {"customer": self.customer_strategy_sha256,
                                "service": self.service_strategy_sha256},
            "output_path": self.output_path, "checkpoint_path": self.checkpoint_path,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())

    def to_document(self) -> dict[str, Any]:
        payload = self.to_payload()
        payload["manifest_sha256"] = self.sha256
        return payload


@dataclass(frozen=True, slots=True)
class PilotManifest:
    """Frozen Phase 4 design; one condition is executed over seed blocks."""

    experiment_id: str
    condition: str
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
    evolution_seeds: tuple[int, ...]
    generations: int
    customer_candidates: int
    max_steps: int
    max_episodes: int
    request_budget_cap: int
    provider_retries: int
    max_concurrency: int
    real_provider_enabled: bool
    role_models: tuple[tuple[str, str | None], ...]
    role_model_args: tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...]
    source_blob_sha1: tuple[tuple[str, str], ...]
    customer_strategy_sha256: str
    service_strategy_sha256: str
    output_path: str
    checkpoint_path: str
    domain: str = "retail"
    communication_mode: str = "half_duplex_text"
    enforce_communication_protocol: bool = False
    evaluation_type: str = "all"
    split_name: str = "train"
    heldout_split_name: str = "test"
    static_customer_portfolio_json: str | None = None
    static_customer_portfolio_sha256: str | None = None
    task_semantic_review_json: str | None = None
    task_semantic_review_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_code_provenance(self.evotau_git_commit, self.evotau_source_sha256)
        if not self.experiment_id.strip():
            raise ValueError("Pilot experiment ID must not be empty")
        if self.condition not in {
            "adaptive_coevolution", "one_shot_repair", "static_customer", "random_mutation",
            "adaptive_customer", "frozen_service", "frozen_customer",
            "no_historical_replay",
        }:
            raise ValueError("Pilot condition is not a supported EvoTau condition")
        if (self.upstream_repository, self.upstream_commit, self.upstream_package_version) != (
            TAU_BENCH_REPOSITORY, TAU_BENCH_COMMIT, TAU2_PACKAGE_VERSION,
        ):
            raise ValueError("Pilot must use the audited tau-bench pin")
        if (self.domain, self.communication_mode, self.evaluation_type) != (
            "retail", "half_duplex_text", "all",
        ):
            raise ValueError("Pilot is fixed to Retail half-duplex text with evaluation_type=all")
        if type(self.enforce_communication_protocol) is not bool:
            raise ValueError("enforce_communication_protocol must be a boolean")
        if self.split_name != "train" or self.heldout_split_name != "test":
            raise ValueError("Pilot E/V/H must use official train/train/test splits")
        if not 4 <= len(self.evolution_task_ids) <= 6:
            raise ValueError("Pilot freezes four to six E tasks")
        if not 2 <= len(self.validation_task_ids) <= 3:
            raise ValueError("Pilot freezes two to three V tasks")
        if not 2 <= len(self.heldout_task_ids) <= 3:
            raise ValueError("Pilot freezes two to three sealed H tasks")
        groups = self.evolution_task_ids + self.validation_task_ids + self.heldout_task_ids
        if any(not isinstance(item, str) or not item.strip() for item in groups):
            raise ValueError("Pilot task IDs must be non-empty strings")
        if len(groups) != len(set(groups)):
            raise ValueError("Pilot E/V/H task IDs must be unique and disjoint")
        if set(groups) & set(self.excluded_task_ids):
            raise ValueError("Pilot cannot select an excluded task")
        if (self.task_semantic_review_json is None
                or self.task_semantic_review_sha256 is None):
            raise ValueError("Pilot requires a frozen ex-ante task semantic review")
        try:
            review_document = json.loads(self.task_semantic_review_json)
            from .task_review import validate_task_semantic_review

            normalized_review, review_sha256 = validate_task_semantic_review(
                review_document,
                expected_panels=(
                    *((task_id, "evolution") for task_id in self.evolution_task_ids),
                    *((task_id, "validation") for task_id in self.validation_task_ids),
                    *((task_id, "heldout") for task_id in self.heldout_task_ids),
                ),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Pilot task semantic review: {exc}") from exc
        if (canonical_json(normalized_review) != self.task_semantic_review_json
                or review_sha256 != self.task_semantic_review_sha256):
            raise ValueError("Pilot task semantic review canonical payload or digest differs")
        if len(self.evolution_seeds) < 3 or len(set(self.evolution_seeds)) != len(self.evolution_seeds):
            raise ValueError("Pilot requires at least three unique evolution seeds")
        if any(type(seed) is not int or seed < 0 for seed in self.evolution_seeds):
            raise ValueError("Pilot seeds must be non-negative integers")
        if self.generations != 3 or self.customer_candidates != 2:
            raise ValueError("Pilot freezes exactly three generations and K=2 Customer candidates")
        if self.max_steps != 64:
            raise ValueError("Pilot max_steps must be 64")
        if type(self.max_episodes) is not int or self.max_episodes <= 0:
            raise ValueError("Pilot max_episodes must be a positive per-seed cap")
        minimum_scheduled_episodes = (
            len(self.evolution_task_ids) * self.generations * (1 + self.customer_candidates)
            + len(self.validation_task_ids) + len(self.heldout_task_ids)
        )
        if self.max_episodes < minimum_scheduled_episodes:
            raise ValueError(
                "Pilot episode cap cannot fit its frozen discovery schedule and final V/H panels"
            )
        if type(self.request_budget_cap) is not int or self.request_budget_cap <= 0:
            raise ValueError("Pilot request_budget_cap must be a positive per-seed cap")
        if self.provider_retries != 0 or type(self.max_concurrency) is not int or not 1 <= self.max_concurrency <= 4:
            raise ValueError("Pilot requires provider retries=0 and concurrency in the range 1..4")
        if type(self.real_provider_enabled) is not bool:
            raise ValueError("Pilot real_provider_enabled must be boolean")
        models = dict(self.role_models)
        if tuple(name for name, _ in self.role_models) != MECHANISM_ROLE_NAMES:
            raise ValueError(f"Pilot models must freeze exactly {sorted(MECHANISM_ROLE_NAMES)}")
        freeze_role_model_args(
            _role_model_args_payload(self.role_model_args), roles=MECHANISM_ROLE_NAMES,
        )
        if self.real_provider_enabled and any(not models[name] for name in models):
            raise ValueError("live Pilot requires frozen model IDs for every role")
        blobs = dict(self.source_blob_sha1)
        missing = REQUIRED_SOURCE_PATHS - set(blobs)
        if missing:
            raise ValueError(f"Pilot is missing upstream source fingerprints: {sorted(missing)}")
        if any(not re.fullmatch(r"[0-9a-f]{40}", digest) for digest in blobs.values()):
            raise ValueError("Pilot source fingerprints must be lowercase Git blob SHA-1 values")
        for name, digest in (
            ("customer", self.customer_strategy_sha256),
            ("service", self.service_strategy_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Pilot {name} strategy hash must be a SHA-256 hex digest")
        if self.condition == "static_customer":
            if self.static_customer_portfolio_json is None or self.static_customer_portfolio_sha256 is None:
                raise ValueError("static_customer Pilot requires a frozen Customer portfolio")
            try:
                portfolio_document = json.loads(self.static_customer_portfolio_json)
                from .baselines import StaticCustomerPortfolio

                portfolio = StaticCustomerPortfolio.from_dict(portfolio_document)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid static Customer portfolio: {exc}") from exc
            if (json.dumps(portfolio.to_dict(), sort_keys=True, separators=(",", ":"))
                    != self.static_customer_portfolio_json
                    or portfolio.sha256 != self.static_customer_portfolio_sha256):
                raise ValueError("static Customer portfolio canonical payload or digest differs")
        elif (self.static_customer_portfolio_json is not None
              or self.static_customer_portfolio_sha256 is not None):
            raise ValueError("static Customer portfolio is only valid for static_customer Pilot")
        _relative_path(self.output_path, "output_path")
        _relative_path(self.checkpoint_path, "checkpoint_path")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PilotManifest:
        experiment = raw["experiment"]
        if experiment.get("phase") != "4-pilot":
            raise ValueError("PilotManifest requires phase='4-pilot'")
        selection = experiment["task_selection"]
        if selection.get("source_split") != "train" or selection.get("heldout_split") != "test":
            raise ValueError("Pilot task_selection must declare train E/V and test H sources")
        if any(not isinstance(selection.get(name), list) for name in ("evolution", "validation", "heldout")):
            raise TypeError("Pilot E/V/H task selections must be JSON arrays")
        selected_ids = tuple(
            str(task_id)
            for name in ("evolution", "validation", "heldout")
            for task_id in selection[name]
        )
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("Pilot E/V/H task IDs must be unique and disjoint")
        seeds = experiment.get("evolution_seeds")
        if not isinstance(seeds, list):
            raise TypeError("Pilot evolution_seeds must be a JSON array")
        upstream = experiment["upstream"]
        code = capture_code_provenance()
        models = experiment.get("models", {})
        customer = experiment.get("customer_strategy")
        service = experiment.get("service_strategy") or {"rules": []}
        if experiment.get("condition") == "frozen_customer" and not isinstance(customer, dict):
            raise ValueError("frozen_customer condition requires an explicit fixed CustomerStrategy")
        expected_panels = (
            *((str(task_id), "evolution") for task_id in selection["evolution"]),
            *((str(task_id), "validation") for task_id in selection["validation"]),
            *((str(task_id), "heldout") for task_id in selection["heldout"]),
        )
        raw_task_review = experiment.get("task_semantic_review")
        from .task_review import validate_task_semantic_review

        normalized_task_review, task_review_sha256 = validate_task_semantic_review(
            raw_task_review, expected_panels=expected_panels,
        )
        task_review_json = canonical_json(normalized_task_review)
        static_portfolio_json = None
        static_portfolio_sha256 = None
        if experiment.get("condition") == "static_customer":
            raw_portfolio = experiment.get("static_customer_portfolio")
            if not isinstance(raw_portfolio, dict):
                raise ValueError("static_customer Pilot requires experiment.static_customer_portfolio")
            from .baselines import StaticCustomerPortfolio

            static_portfolio = StaticCustomerPortfolio.from_dict(raw_portfolio)
            static_portfolio_json = json.dumps(
                static_portfolio.to_dict(), sort_keys=True, separators=(",", ":"),
            )
            static_portfolio_sha256 = static_portfolio.sha256
        elif "static_customer_portfolio" in experiment:
            raise ValueError("static_customer_portfolio is only valid for static_customer Pilot")
        return cls(
            experiment_id=str(experiment["id"]),
            condition=str(experiment["condition"]),
            upstream_repository=str(upstream["repository"]),
            upstream_commit=str(upstream["commit"]),
            upstream_package_version=str(upstream["package_version"]),
            evotau_git_commit=code.git_commit,
            evotau_working_tree_clean=code.working_tree_clean,
            evotau_source_sha256=code.source_sha256,
            evolution_task_ids=tuple(str(item) for item in selection["evolution"]),
            validation_task_ids=tuple(str(item) for item in selection["validation"]),
            heldout_task_ids=tuple(str(item) for item in selection["heldout"]),
            excluded_task_ids=tuple(str(item) for item in selection.get("excluded", ())),
            evolution_seeds=tuple(seeds),
            generations=int(experiment["generations"]),
            customer_candidates=int(experiment.get("customer_candidates", 2)),
            max_steps=int(experiment["max_steps"]),
            max_episodes=int(experiment["max_episodes"]),
            request_budget_cap=int(experiment["request_budget_cap"]),
            provider_retries=int(experiment["provider_retries"]),
            max_concurrency=int(experiment["max_concurrency"]),
            real_provider_enabled=experiment["real_provider_enabled"],
            role_models=_freeze_role_models(models, roles=MECHANISM_ROLE_NAMES),
            role_model_args=freeze_role_model_args(
                experiment.get("model_args"), roles=MECHANISM_ROLE_NAMES,
            ),
            source_blob_sha1=tuple(sorted(
                (str(path), str(digest).lower())
                for path, digest in experiment["source_blob_sha1"].items()
            )),
            customer_strategy_sha256=sha256_json(
                {"disclosure": "minimal_on_request", "request_order": "scenario_order",
                 "challenge_style": "none", "challenge_budget": 0}
                if customer is None else customer
            ),
            service_strategy_sha256=sha256_json(service),
            output_path=_relative_path(str(experiment["output_path"]), "output_path"),
            checkpoint_path=_relative_path(str(experiment["checkpoint_path"]), "checkpoint_path"),
            heldout_split_name=str(selection["heldout_split"]),
            static_customer_portfolio_json=static_portfolio_json,
            static_customer_portfolio_sha256=static_portfolio_sha256,
            task_semantic_review_json=task_review_json,
            task_semantic_review_sha256=task_review_sha256,
            enforce_communication_protocol=_communication_protocol_enforcement(experiment),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "experiment_id": self.experiment_id,
            "phase": "4-pilot",
            "condition": self.condition,
            "upstream": {
                "repository": self.upstream_repository, "commit": self.upstream_commit,
                "package_version": self.upstream_package_version,
            },
            "evotau": {
                "git_commit": self.evotau_git_commit,
                "working_tree_clean": self.evotau_working_tree_clean,
                "source_sha256": self.evotau_source_sha256,
            },
            "domain": self.domain, "communication_mode": self.communication_mode,
            "enforce_communication_protocol": self.enforce_communication_protocol,
            "evaluation_type": self.evaluation_type,
            "task_selection": {
                "source_split": self.split_name,
                "evolution": list(self.evolution_task_ids),
                "validation": list(self.validation_task_ids),
                "heldout_split": self.heldout_split_name,
                "heldout": list(self.heldout_task_ids),
                "excluded": list(self.excluded_task_ids),
            },
            "evolution_seeds": list(self.evolution_seeds),
            "generations": self.generations,
            "customer_candidates": self.customer_candidates,
            "max_steps": self.max_steps,
            "max_episodes": self.max_episodes,
            "request_budget_cap": self.request_budget_cap,
            "provider_retries": self.provider_retries,
            "max_concurrency": self.max_concurrency,
            "real_provider_enabled": self.real_provider_enabled,
            "role_models": dict(self.role_models),
            "role_model_args": _role_model_args_payload(self.role_model_args),
            "runtime_arguments": _role_runtime_arguments(self.role_model_args),
            "source_blob_sha1": dict(self.source_blob_sha1),
            "failure_taxonomy": [
                {
                    "workflow_stage": stage,
                    "policy_rule_id": policy_rule_id,
                    "mistake_type": mistake,
                }
                for stage, policy_rule_id, mistake in MVP_FAILURE_TAXONOMY
            ],
            "failure_taxonomy_sha256": sha256_json(MVP_FAILURE_TAXONOMY),
            "strategy_sha256": {
                "customer": self.customer_strategy_sha256,
                "service": self.service_strategy_sha256,
            },
            "paths": {"output": self.output_path, "checkpoint": self.checkpoint_path},
        }
        if self.static_customer_portfolio_json is not None:
            payload["static_customer_portfolio"] = json.loads(self.static_customer_portfolio_json)
            payload["static_customer_portfolio_sha256"] = self.static_customer_portfolio_sha256
        payload["task_semantic_review"] = json.loads(self.task_semantic_review_json)
        payload["task_semantic_review_sha256"] = self.task_semantic_review_sha256
        return payload

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())

    def to_document(self) -> dict[str, Any]:
        payload = self.to_payload()
        payload["manifest_sha256"] = self.sha256
        return payload


@dataclass(frozen=True, slots=True)
class ActivationSmokeManifest:
    """Independent one-generation, ten-task Service-repair activation protocol."""

    experiment_id: str
    upstream_repository: str
    upstream_commit: str
    upstream_package_version: str
    evotau_git_commit: str | None
    evotau_working_tree_clean: bool | None
    evotau_source_sha256: str
    evolution_task_ids: tuple[str, ...]
    validation_task_ids: tuple[str, ...]
    excluded_task_ids: tuple[str, ...]
    seed: int
    max_steps: int
    max_episodes: int
    request_budget_cap: int | None
    provider_retries: int
    max_concurrency: int
    real_provider_enabled: bool
    role_models: tuple[tuple[str, str | None], ...]
    role_model_args: tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...]
    source_blob_sha1: tuple[tuple[str, str], ...]
    customer_strategy_sha256: str
    service_strategy_sha256: str
    task_review_path: str
    task_semantic_review_json: str
    task_semantic_review_sha256: str
    domain: str = "retail"
    communication_mode: str = "half_duplex_text"
    enforce_communication_protocol: bool = False
    evaluation_type: str = "all"
    split_name: str = "train"
    generations: int = 1
    customer_candidates: int = 2
    condition: str = "adaptive_coevolution"
    output_path: str = "experiments/runs/evotau-activation-smoke"
    checkpoint_path: str = "experiments/checkpoints/evotau-activation-smoke"
    unbounded_provider_budget: bool = False
    customer_evolver_schema: str = "operator_v1"

    def __post_init__(self) -> None:
        _validate_code_provenance(self.evotau_git_commit, self.evotau_source_sha256)
        if not self.experiment_id.strip():
            raise ValueError("activation smoke experiment ID must not be empty")
        if (self.upstream_repository, self.upstream_commit, self.upstream_package_version) != (
            TAU_BENCH_REPOSITORY, TAU_BENCH_COMMIT, TAU2_PACKAGE_VERSION,
        ):
            raise ValueError("activation smoke must use the audited tau-bench pin")
        if (self.domain, self.communication_mode, self.evaluation_type, self.split_name) != (
            "retail", "half_duplex_text", "all", "train",
        ):
            raise ValueError("activation smoke is fixed to Retail, half-duplex text, and train E/V")
        if type(self.enforce_communication_protocol) is not bool or self.enforce_communication_protocol:
            raise ValueError("activation smoke records communication protocol without enforcement")
        if self.condition != "adaptive_coevolution":
            raise ValueError("activation smoke freezes the existing adaptive co-evolution condition")
        if self.customer_evolver_schema not in {"operator_v1", "strategy_v2"}:
            raise ValueError("activation Customer Evolver schema must be operator_v1 or strategy_v2")
        if len(self.evolution_task_ids) != 10 or len(set(self.evolution_task_ids)) != 10:
            raise ValueError("activation smoke requires exactly ten unique E tasks")
        if not self.validation_task_ids or len(set(self.validation_task_ids)) != len(self.validation_task_ids):
            raise ValueError("activation smoke requires a non-empty unique validation panel")
        all_ids = self.evolution_task_ids + self.validation_task_ids
        if any(not item.strip() for item in all_ids) or len(set(all_ids)) != len(all_ids):
            raise ValueError("activation E/V task IDs must be non-empty and disjoint")
        if set(all_ids) & set(self.excluded_task_ids):
            raise ValueError("activation E/V panels cannot contain explicitly excluded tasks")
        if self.seed != 1 or self.generations != 1 or self.customer_candidates != 2:
            raise ValueError("activation smoke freezes seed=1, generations=1, and K=2")
        if self.max_steps != 64:
            raise ValueError("activation smoke max_steps must be 64")
        if type(self.max_episodes) is not int or not 30 <= self.max_episodes <= 100:
            raise ValueError("activation smoke episode cap must cover 30 discovery episodes and be at most 100")
        if type(self.unbounded_provider_budget) is not bool:
            raise ValueError("unbounded provider budget flag must be boolean")
        if self.unbounded_provider_budget:
            if self.request_budget_cap is not None:
                raise ValueError("unbounded provider budget must not declare a request cap")
        elif type(self.request_budget_cap) is not int or not 1 <= self.request_budget_cap <= 1800:
            raise ValueError("activation smoke request cap must be in the range 1..1800")
        if self.provider_retries != 0 or type(self.max_concurrency) is not int or not 1 <= self.max_concurrency <= 4:
            raise ValueError("activation smoke requires retries=0 and concurrency in the range 1..4")
        if type(self.real_provider_enabled) is not bool:
            raise ValueError("activation real_provider_enabled must be boolean")
        models = dict(self.role_models)
        if tuple(name for name, _ in self.role_models) != MECHANISM_ROLE_NAMES:
            raise ValueError(f"activation models must freeze exactly {sorted(MECHANISM_ROLE_NAMES)}")
        if self.real_provider_enabled and any(not models[name] for name in MECHANISM_ROLE_NAMES):
            raise ValueError("live activation smoke requires frozen model IDs for every role")
        model_args_payload = _role_model_args_payload(self.role_model_args)
        freeze_role_model_args(model_args_payload, roles=MECHANISM_ROLE_NAMES)
        if self.unbounded_provider_budget and any(
            "max_tokens" in values for values in model_args_payload.values()
        ):
            raise ValueError("unbounded provider configuration must omit max_tokens for every role")
        blobs = dict(self.source_blob_sha1)
        missing = REQUIRED_SOURCE_PATHS - set(blobs)
        if missing:
            raise ValueError(f"activation smoke is missing upstream source fingerprints: {sorted(missing)}")
        if any(not re.fullmatch(r"[0-9a-f]{40}", digest) for digest in blobs.values()):
            raise ValueError("activation source fingerprints must be lowercase Git blob SHA-1 values")
        for name, digest in (
            ("customer", self.customer_strategy_sha256),
            ("service", self.service_strategy_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"activation {name} strategy hash must be a SHA-256 hex digest")
        _relative_path(self.task_review_path, "task_review_path")
        _relative_path(self.output_path, "output_path")
        _relative_path(self.checkpoint_path, "checkpoint_path")
        try:
            review = json.loads(self.task_semantic_review_json)
            from .task_review import validate_task_semantic_review

            normalized, digest = validate_task_semantic_review(
                review,
                expected_panels=(
                    *((task_id, "evolution") for task_id in self.evolution_task_ids),
                    *((task_id, "validation") for task_id in self.validation_task_ids),
                ),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid activation human task review: {exc}") from exc
        if canonical_json(normalized) != self.task_semantic_review_json or digest != self.task_semantic_review_sha256:
            raise ValueError("activation human task review payload or digest differs")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        task_review_document: Mapping[str, Any],
    ) -> ActivationSmokeManifest:
        experiment = raw["experiment"]
        selection = experiment["task_selection"]
        upstream = experiment["upstream"]
        if experiment.get("phase") != "3-multitask-service-repair-activation":
            raise ValueError("ActivationSmokeManifest requires the multi-task activation phase")
        if selection.get("source_split") != "train" or selection.get("heldout", ()):
            raise ValueError("activation E/V panels must be train-only and keep H empty")
        validation_ids = tuple(str(item) for item in selection.get("validation", ()))
        if not validation_ids:
            raise ValueError("activation config must retain an explicit reviewed validation panel")
        if selection.get("evolution", ()):
            raise ValueError("activation E tasks come only from the attached human-reviewed task panel")
        if not isinstance(task_review_document, Mapping):
            raise TypeError("activation requires the existing task-review schema as an input artifact")
        review_rows = task_review_document.get("task_reviews")
        if not isinstance(review_rows, list):
            raise TypeError("activation task review must contain task_reviews")
        evolution_ids = tuple(
            str(row.get("task_id")) for row in review_rows
            if isinstance(row, Mapping) and row.get("panel") == "evolution"
        )
        reviewed_validation = tuple(
            str(row.get("task_id")) for row in review_rows
            if isinstance(row, Mapping) and row.get("panel") == "validation"
        )
        if reviewed_validation != validation_ids:
            raise ValueError("activation validation task IDs differ from the human-reviewed task panel")
        from .task_review import validate_task_semantic_review

        expected_panels = (
            *((task_id, "evolution") for task_id in evolution_ids),
            *((task_id, "validation") for task_id in validation_ids),
        )
        normalized_review, review_sha256 = validate_task_semantic_review(
            dict(task_review_document), expected_panels=expected_panels,
        )
        models = experiment.get("models", {})
        diagnostic_capture = raw.get("diagnostic_capture", {})
        configured_unbounded = experiment.get("unbounded_provider_budget", False)
        if type(configured_unbounded) is not bool:
            raise ValueError("unbounded_provider_budget must be boolean")
        unbounded_provider_budget = configured_unbounded or (
            isinstance(diagnostic_capture, Mapping)
            and diagnostic_capture.get("unbounded_request_budget") is True
        )
        request_budget_cap = experiment.get("request_budget_cap")
        if request_budget_cap is None and not unbounded_provider_budget:
            raise ValueError("an uncapped request budget requires explicit configuration")
        if unbounded_provider_budget and request_budget_cap is not None:
            raise ValueError("unbounded provider configuration must set request_budget_cap to null")
        customer = experiment.get("customer_strategy")
        customer_evolver_schema = experiment.get("customer_evolver_schema", "operator_v1")
        if customer_evolver_schema not in {"operator_v1", "strategy_v2"}:
            raise ValueError("unsupported activation Customer Evolver schema")
        if customer_evolver_schema == "strategy_v2" and not isinstance(customer, Mapping):
            raise ValueError("strategy_v2 requires an explicit seven-field Customer incumbent")
        if customer is not None:
            from .strategies import CustomerStrategy

            parsed_customer = CustomerStrategy(**customer)
            if parsed_customer.is_v2 != (customer_evolver_schema == "strategy_v2"):
                raise ValueError("activation Customer strategy and Evolver schema differ")
        service = experiment.get("service_strategy") or {"rules": []}
        code = capture_code_provenance()
        return cls(
            experiment_id=str(experiment["id"]),
            upstream_repository=str(upstream["repository"]),
            upstream_commit=str(upstream["commit"]),
            upstream_package_version=str(upstream["package_version"]),
            evotau_git_commit=code.git_commit,
            evotau_working_tree_clean=code.working_tree_clean,
            evotau_source_sha256=code.source_sha256,
            evolution_task_ids=evolution_ids,
            validation_task_ids=validation_ids,
            excluded_task_ids=tuple(str(item) for item in selection.get("excluded", ())),
            seed=int(experiment["seed"]),
            max_steps=int(experiment["max_steps"]),
            max_episodes=int(experiment["max_episodes"]),
            request_budget_cap=(
                None if request_budget_cap is None else int(request_budget_cap)
            ),
            provider_retries=int(experiment["provider_retries"]),
            max_concurrency=int(experiment["max_concurrency"]),
            real_provider_enabled=experiment["real_provider_enabled"],
            role_models=_freeze_role_models(models, roles=MECHANISM_ROLE_NAMES),
            role_model_args=freeze_role_model_args(experiment.get("model_args"), roles=MECHANISM_ROLE_NAMES),
            source_blob_sha1=tuple(sorted(
                (str(path), str(digest).lower())
                for path, digest in experiment["source_blob_sha1"].items()
            )),
            customer_strategy_sha256=(
                DEFAULT_CUSTOMER_STRATEGY_HASH if customer is None else sha256_json(customer)
            ),
            service_strategy_sha256=sha256_json(service),
            task_review_path=str(experiment["task_review_path"]),
            task_semantic_review_json=canonical_json(normalized_review),
            task_semantic_review_sha256=review_sha256,
            domain=str(experiment["domain"]),
            communication_mode=str(experiment["communication_mode"]),
            enforce_communication_protocol=_communication_protocol_enforcement(experiment),
            evaluation_type=str(experiment["evaluation_type"]),
            split_name=str(selection["source_split"]),
            generations=int(experiment["generations"]),
            customer_candidates=int(experiment["customer_candidates"]),
            condition=str(experiment["condition"]),
            output_path=_relative_path(str(experiment["output_path"]), "output_path"),
            checkpoint_path=_relative_path(str(experiment["checkpoint_path"]), "checkpoint_path"),
            unbounded_provider_budget=unbounded_provider_budget,
            customer_evolver_schema=customer_evolver_schema,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "experiment_id": self.experiment_id,
            "phase": "3-multitask-service-repair-activation",
            "condition": self.condition,
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
                "heldout": [],
                "excluded": list(self.excluded_task_ids),
            },
            "seed": self.seed,
            "evolution_seeds": [self.seed],
            "generations": self.generations,
            "customer_candidates": self.customer_candidates,
            "max_steps": self.max_steps,
            "max_episodes": self.max_episodes,
            "request_budget_cap": self.request_budget_cap,
            "unbounded_provider_budget": self.unbounded_provider_budget,
            "provider_retries": self.provider_retries,
            "max_concurrency": self.max_concurrency,
            "real_provider_enabled": self.real_provider_enabled,
            "role_models": dict(self.role_models),
            "role_model_args": _role_model_args_payload(self.role_model_args),
            "runtime_arguments": _role_runtime_arguments(self.role_model_args),
            "source_blob_sha1": dict(self.source_blob_sha1),
            "failure_taxonomy": [
                {"workflow_stage": stage, "policy_rule_id": rule_id, "mistake_type": mistake}
                for stage, rule_id, mistake in MVP_FAILURE_TAXONOMY
            ],
            "failure_taxonomy_sha256": sha256_json(MVP_FAILURE_TAXONOMY),
            "strategy_sha256": {
                "customer": self.customer_strategy_sha256,
                "service": self.service_strategy_sha256,
            },
            "task_review_path": self.task_review_path,
            "task_semantic_review": json.loads(self.task_semantic_review_json),
            "task_semantic_review_sha256": self.task_semantic_review_sha256,
            "auditor_calibration_status": "role-separated_but_not_yet_human-calibrated",
            "paths": {"output": self.output_path, "checkpoint": self.checkpoint_path},
        }
        if self.customer_evolver_schema == "strategy_v2":
            payload["customer_evolver_schema"] = "strategy_v2"
        return payload

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())

    def to_document(self) -> dict[str, Any]:
        payload = self.to_payload()
        payload["manifest_sha256"] = self.sha256
        return payload


def write_manifest_once(
    path: str | Path,
    manifest: ExperimentManifest | MechanismManifest | PilotManifest | ActivationSmokeManifest,
) -> Path:
    """Write an immutable JSON manifest; never replace an existing artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest.to_document(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target
