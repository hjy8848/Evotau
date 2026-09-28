"""Small, serializable records shared by evolution, evaluation and replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from .manifest import sha256_json
from .strategies import CustomerStrategy, ServiceStrategy


class EpisodeStatus(StrEnum):
    COMPLETE = "complete"
    INVALID_CUSTOMER = "invalid_customer"
    INVALID_STRATEGY = "invalid_strategy"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A pointer into a native trajectory, not a copied synthetic story."""

    turn_index: int
    source: str
    summary: str

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError("evidence turn_index must be non-negative")
        if self.source not in {"user", "assistant", "tool", "evaluator", "reviewer"}:
            raise ValueError(f"unsupported evidence source: {self.source}")
        if not self.summary.strip():
            raise ValueError("evidence summary must not be empty")

    def stable_ref(self, episode_id: str) -> str:
        """Return a trajectory pointer that does not copy potentially sensitive text."""
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        return f"{episode_id}#turn:{self.turn_index}:{self.source}"


@dataclass(frozen=True, slots=True)
class FailureSignature:
    domain: str
    workflow_stage: str
    policy_rule_id: str
    mistake_type: str

    def __post_init__(self) -> None:
        if not all((self.domain, self.workflow_stage, self.policy_rule_id, self.mistake_type)):
            raise ValueError("all failure signature components must be non-empty")

    @property
    def key(self) -> str:
        return sha256_json(asdict(self))[:16]

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One executed or fixture episode and the native evidence attached to it."""

    episode_id: str
    task_id: str
    seed: int
    customer_strategy_id: str
    service_strategy_id: str
    status: EpisodeStatus
    task_success: bool | None
    native_reward: float | None = None
    termination_reason: str | None = None
    customer_valid: bool | None = None
    customer_strategy_adherent: bool | None = None
    policy_violation: bool = False
    policy_rule_id: str | None = None
    mistake_type: str | None = None
    workflow_stage: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    trajectory_ref: str | None = None
    tool_calls: int = 0
    raw_review: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.episode_id or not self.task_id:
            raise ValueError("episode_id and task_id must be non-empty")
        if self.seed < 0:
            raise ValueError("episode seed must be non-negative")
        if self.tool_calls < 0:
            raise ValueError("tool_calls must be non-negative")
        if self.status == EpisodeStatus.COMPLETE and self.task_success is None:
            raise ValueError("a complete episode must record task_success")

    @property
    def episode_key(self) -> str:
        return sha256_json(
            {
                "episode_id": self.episode_id,
                "task_id": self.task_id,
                "seed": self.seed,
                "customer": self.customer_strategy_id,
                "service": self.service_strategy_id,
            }
        )

    @property
    def has_attributable_failure_candidate(self) -> bool:
        return bool(
            self.status == EpisodeStatus.COMPLETE
            and self.customer_valid is True
            and self.customer_strategy_adherent is True
            and self.policy_violation
            and self.policy_rule_id
            and self.mistake_type
            and self.workflow_stage
            and self.evidence
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EpisodeRecord:
        fields = dict(value)
        fields["status"] = EpisodeStatus(fields["status"])
        fields["evidence"] = tuple(EvidenceRef(**item) for item in fields.get("evidence", ()))
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """A human or protocol verified service failure, safe to drive evolution."""

    failure_id: str
    episode_id: str
    task_id: str
    generation: int
    customer_strategy_id: str
    service_strategy_id: str
    signature: FailureSignature
    policy_ref: str
    evidence: tuple[EvidenceRef, ...]
    verification_ref: str
    severity: str = "material"

    def __post_init__(self) -> None:
        if not all((self.failure_id, self.episode_id, self.task_id, self.policy_ref, self.verification_ref)):
            raise ValueError("verified failure identifiers and policy/verification refs are required")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if not self.evidence:
            raise ValueError("verified failure must cite trajectory evidence")
        if self.signature.policy_rule_id != self.policy_ref:
            raise ValueError("failure signature and policy reference must agree")

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return tuple(ref.stable_ref(self.episode_id) for ref in self.evidence)

    @classmethod
    def verify(
        cls,
        episode: EpisodeRecord,
        *,
        generation: int,
        verifier: str,
    ) -> FailureRecord:
        """Promote a fully evidenced candidate after an explicit independent audit."""

        if not episode.has_attributable_failure_candidate:
            raise ValueError("episode lacks complete valid, adherent, policy-linked failure evidence")
        assert episode.policy_rule_id and episode.mistake_type and episode.workflow_stage
        signature = FailureSignature(
            "retail", episode.workflow_stage, episode.policy_rule_id, episode.mistake_type
        )
        failure_id = sha256_json(
            {"episode": episode.episode_key, "signature": signature.to_dict(), "verifier": verifier}
        )[:20]
        return cls(
            failure_id=failure_id,
            episode_id=episode.episode_id,
            task_id=episode.task_id,
            generation=generation,
            customer_strategy_id=episode.customer_strategy_id,
            service_strategy_id=episode.service_strategy_id,
            signature=signature,
            policy_ref=episode.policy_rule_id,
            evidence=episode.evidence,
            verification_ref=verifier,
        )


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    strategy_id: str
    episodes: tuple[EpisodeRecord, ...]
    verified_failures: tuple[FailureRecord, ...] = ()
    panel_name: str = "discovery"

    @property
    def fitness(self) -> int:
        """Count distinct failed tasks, with at most one point per task."""

        return len({failure.task_id for failure in self.verified_failures})

    @property
    def signature_count(self) -> int:
        return len({failure.signature.key for failure in self.verified_failures})

    @property
    def valid_episode_count(self) -> int:
        return sum(item.status == EpisodeStatus.COMPLETE for item in self.episodes)

    @property
    def invalid_episode_count(self) -> int:
        return len(self.episodes) - self.valid_episode_count

    @property
    def task_coverage(self) -> int:
        return len({item.task_id for item in self.episodes})

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "panel_name": self.panel_name,
            "fitness": self.fitness,
            "signature_count": self.signature_count,
            "valid_episode_count": self.valid_episode_count,
            "invalid_episode_count": self.invalid_episode_count,
            "episodes": [item.to_dict() for item in self.episodes],
            "failure_ids": [item.failure_id for item in self.verified_failures],
        }


def customer_strategy_id(strategy: CustomerStrategy) -> str:
    return sha256_json(strategy.to_dict())[:16]


def service_strategy_id(strategy: ServiceStrategy) -> str:
    return sha256_json(strategy.to_dict())[:16]
