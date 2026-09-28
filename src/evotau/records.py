"""Small, serializable records shared by evolution, evaluation and replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from .manifest import MVP_FAILURE_TAXONOMY, sha256_json
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


def is_mvp_failure_signature(signature: FailureSignature) -> bool:
    return signature.domain == "retail" and (
        signature.workflow_stage, signature.policy_rule_id, signature.mistake_type
    ) in MVP_FAILURE_TAXONOMY


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
    strategy_applicable: bool | None = None
    customer_strategy_adherent: bool | None = None
    policy_violation: bool = False
    invalid_repeated_write_calls: int | None = None
    policy_rule_id: str | None = None
    mistake_type: str | None = None
    workflow_stage: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    trajectory_ref: str | None = None
    audit_ref: str | None = None
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
        if (self.invalid_repeated_write_calls is not None
                and (type(self.invalid_repeated_write_calls) is not int
                     or self.invalid_repeated_write_calls < 0)):
            raise ValueError("invalid_repeated_write_calls must be a non-negative integer or None")
        for name in ("customer_valid", "strategy_applicable", "customer_strategy_adherent"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"{name} must be bool or None")
        if self.strategy_applicable is False and self.customer_strategy_adherent is not None:
            raise ValueError("not_applicable Customer strategy behavior cannot have an adherence judgment")
        if self.strategy_applicable is True and self.customer_strategy_adherent is None:
            raise ValueError("applicable Customer strategy behavior requires an adherence judgment")
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
            and self.strategy_applicable is not False
            and self.customer_strategy_adherent is True
            and self.policy_violation
            and self.policy_rule_id
            and self.mistake_type
            and self.workflow_stage
            and self.evidence
        )

    @property
    def strategy_opportunity(self) -> bool:
        return self.strategy_applicable is True or (
            self.strategy_applicable is None and self.customer_strategy_adherent is not None
        )

    @property
    def strategy_not_applicable(self) -> bool:
        return self.strategy_applicable is False

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
    reproduction_episode_id: str
    reproduction_verification_ref: str
    severity: str = "material"

    def __post_init__(self) -> None:
        if not all((
            self.failure_id, self.episode_id, self.task_id, self.policy_ref,
            self.verification_ref, self.reproduction_episode_id,
            self.reproduction_verification_ref,
        )):
            raise ValueError("verified failure identifiers and policy/verification refs are required")
        if self.episode_id == self.reproduction_episode_id:
            raise ValueError("failure reproduction must reference a distinct episode")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if not self.evidence:
            raise ValueError("verified failure must cite trajectory evidence")
        if self.signature.policy_rule_id != self.policy_ref:
            raise ValueError("failure signature and policy reference must agree")
        if not is_mvp_failure_signature(self.signature):
            raise ValueError("failure signature is outside the frozen Retail MVP taxonomy")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FailureRecord:
        fields = dict(value)
        fields["signature"] = FailureSignature(**fields["signature"])
        fields["evidence"] = tuple(EvidenceRef(**item) for item in fields.get("evidence", ()))
        return cls(**fields)

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return tuple(ref.stable_ref(self.episode_id) for ref in self.evidence)

    @classmethod
    def verify(
        cls,
        episode: EpisodeRecord,
        *,
        reproduction_episode: EpisodeRecord,
        generation: int,
        verifier: str,
        reproduction_verifier: str,
    ) -> FailureRecord:
        """Promote only when independent audits confirm a matching fresh-seed rerun."""

        if not episode.has_attributable_failure_candidate:
            raise ValueError("episode lacks complete valid, adherent, policy-linked failure evidence")
        if not reproduction_episode.has_attributable_failure_candidate:
            raise ValueError("reproduction episode lacks complete valid, adherent failure evidence")
        if (not isinstance(verifier, str) or not verifier.strip()
                or not isinstance(reproduction_verifier, str) or not reproduction_verifier.strip()):
            raise ValueError("source and reproduction episodes each require an independent audit reference")
        if (episode.task_id, episode.customer_strategy_id, episode.service_strategy_id) != (
            reproduction_episode.task_id,
            reproduction_episode.customer_strategy_id,
            reproduction_episode.service_strategy_id,
        ):
            raise ValueError("failure reproduction must keep task and both strategies fixed")
        if episode.seed == reproduction_episode.seed:
            raise ValueError("failure reproduction must use a fresh seed")
        if (episode.policy_rule_id, episode.workflow_stage, episode.mistake_type) != (
            reproduction_episode.policy_rule_id,
            reproduction_episode.workflow_stage,
            reproduction_episode.mistake_type,
        ):
            raise ValueError("failure reproduction must match the exact verified signature")
        assert episode.policy_rule_id and episode.mistake_type and episode.workflow_stage
        signature = FailureSignature(
            "retail", episode.workflow_stage, episode.policy_rule_id, episode.mistake_type
        )
        if not is_mvp_failure_signature(signature):
            raise ValueError("episode failure is outside the frozen Retail MVP taxonomy")
        failure_id = sha256_json(
            {
                "episode": episode.episode_key,
                "reproduction_episode": reproduction_episode.episode_key,
                "signature": signature.to_dict(),
                "verifier": verifier.strip(),
                "reproduction_verifier": reproduction_verifier.strip(),
            }
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
            verification_ref=verifier.strip(),
            reproduction_episode_id=reproduction_episode.episode_id,
            reproduction_verification_ref=reproduction_verifier.strip(),
        )


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    strategy_id: str
    episodes: tuple[EpisodeRecord, ...]
    verified_failures: tuple[FailureRecord, ...] = ()
    panel_name: str = "discovery"
    failure_audit_refs: tuple[tuple[str, str], ...] = ()
    replication_episodes: tuple[EpisodeRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("candidate evaluation requires a strategy ID")
        episode_ids = [episode.episode_id for episode in self.episodes]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("candidate evaluation cannot contain duplicate episode IDs")
        by_id = {episode.episode_id: episode for episode in self.episodes}
        replication_ids = [episode.episode_id for episode in self.replication_episodes]
        if len(replication_ids) != len(set(replication_ids)) or set(replication_ids) & set(by_id):
            raise ValueError("failure replication episodes must be unique and separate from the scored panel")
        evidence_by_id = by_id | {episode.episode_id: episode for episode in self.replication_episodes}
        audit_episode_ids = [episode_id for episode_id, _ref in self.failure_audit_refs]
        if len(audit_episode_ids) != len(set(audit_episode_ids)):
            raise ValueError("failure audit references must be unique per episode")
        for episode_id, ref in self.failure_audit_refs:
            episode = evidence_by_id.get(episode_id)
            if (episode is None or not isinstance(ref, str) or not ref.strip()
                    or not episode.has_attributable_failure_candidate):
                raise ValueError("failure audit reference must name an independently reviewed failure episode")
        for episode in self.episodes:
            if episode.customer_strategy_id != self.strategy_id:
                raise ValueError("episode Customer strategy does not match its candidate evaluation")
        for failure in self.verified_failures:
            episode = by_id.get(failure.episode_id)
            if episode is None:
                raise ValueError("verified failure must refer to an episode in this evaluation")
            if not episode.has_attributable_failure_candidate:
                raise ValueError("verified failure episode is not a complete valid failure candidate")
            if (
                failure.task_id != episode.task_id
                or failure.customer_strategy_id != episode.customer_strategy_id
                or failure.service_strategy_id != episode.service_strategy_id
                or failure.policy_ref != episode.policy_rule_id
                or failure.signature.workflow_stage != episode.workflow_stage
                or failure.signature.mistake_type != episode.mistake_type
                or failure.evidence != episode.evidence
            ):
                raise ValueError("verified failure fields do not match their episode evidence")
            reproduction_episode = evidence_by_id.get(failure.reproduction_episode_id)
            if reproduction_episode is None:
                raise ValueError("verified failure reproduction must be present in the evaluation evidence")
            if reproduction_episode.episode_id not in replication_ids:
                raise ValueError("verified failure reproduction must be retained outside the scored panel")
            if (reproduction_episode.task_id, reproduction_episode.customer_strategy_id,
                    reproduction_episode.service_strategy_id, reproduction_episode.seed) == (
                episode.task_id, episode.customer_strategy_id, episode.service_strategy_id, episode.seed,
            ):
                raise ValueError("failure reproduction must be a distinct-seed strategy-level rerun")
            audit_refs = dict(self.failure_audit_refs)
            if (audit_refs.get(episode.episode_id) != failure.verification_ref
                    or audit_refs.get(reproduction_episode.episode_id)
                    != failure.reproduction_verification_ref):
                raise ValueError("verified failure audit refs must match both reviewed episodes")

    @property
    def fitness(self) -> int:
        """Count distinct failed tasks, with at most one point per task."""

        return len({failure.task_id for failure in self.verified_failures})

    @property
    def provisional_failure_events(self) -> frozenset[tuple[str, str]]:
        """Audited single-episode signals used only to choose a replay probe."""
        audited = dict(self.failure_audit_refs)
        events: set[tuple[str, str]] = set()
        for episode in self.episodes:
            if episode.episode_id not in audited or not episode.has_attributable_failure_candidate:
                continue
            signature = FailureSignature(
                "retail", episode.workflow_stage or "", episode.policy_rule_id or "",
                episode.mistake_type or "",
            )
            if is_mvp_failure_signature(signature):
                events.add((episode.task_id, signature.key))
        return frozenset(events)

    @property
    def provisional_failure_count(self) -> int:
        """Distinct task count before replay; never used as Customer fitness."""
        return len({task_id for task_id, _signature in self.provisional_failure_events})

    @property
    def signature_count(self) -> int:
        return len({failure.signature.key for failure in self.verified_failures})

    @property
    def valid_episode_count(self) -> int:
        return sum(
            item.status in {EpisodeStatus.COMPLETE, EpisodeStatus.INVALID_STRATEGY}
            and item.customer_valid is True
            for item in self.episodes
        )

    @property
    def invalid_episode_count(self) -> int:
        return sum(
            item.status == EpisodeStatus.INVALID_CUSTOMER or item.customer_valid is False
            for item in self.episodes
        )

    @property
    def uncertain_episode_count(self) -> int:
        return len(self.episodes) - self.valid_episode_count - self.invalid_episode_count

    @property
    def task_coverage(self) -> int:
        return len({item.task_id for item in self.episodes})

    @property
    def strategy_opportunity_count(self) -> int:
        return sum(
            item.status in {EpisodeStatus.COMPLETE, EpisodeStatus.INVALID_STRATEGY}
            and item.customer_valid is True and item.strategy_opportunity
            for item in self.episodes
        )

    @property
    def strategy_adherent_count(self) -> int:
        return sum(
            item.status in {EpisodeStatus.COMPLETE, EpisodeStatus.INVALID_STRATEGY}
            and item.customer_valid is True and item.customer_strategy_adherent is True
            for item in self.episodes
        )

    @property
    def strategy_not_applicable_count(self) -> int:
        return sum(
            item.status in {EpisodeStatus.COMPLETE, EpisodeStatus.INVALID_STRATEGY}
            and item.customer_valid is True and item.strategy_not_applicable
            for item in self.episodes
        )

    @property
    def strategy_adherence_rate(self) -> float | None:
        if not self.strategy_opportunity_count:
            return None
        return self.strategy_adherent_count / self.strategy_opportunity_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "panel_name": self.panel_name,
            "fitness": self.fitness,
            "provisional_failure_count": self.provisional_failure_count,
            "provisional_failure_events": [list(item) for item in sorted(self.provisional_failure_events)],
            "signature_count": self.signature_count,
            "valid_episode_count": self.valid_episode_count,
            "invalid_episode_count": self.invalid_episode_count,
            "uncertain_episode_count": self.uncertain_episode_count,
            "strategy_opportunity_count": self.strategy_opportunity_count,
            "strategy_adherent_count": self.strategy_adherent_count,
            "strategy_not_applicable_count": self.strategy_not_applicable_count,
            "strategy_adherence_rate": self.strategy_adherence_rate,
            "episodes": [item.to_dict() for item in self.episodes],
            "replication_episodes": [item.to_dict() for item in self.replication_episodes],
            "failure_ids": [item.failure_id for item in self.verified_failures],
            "failure_audit_refs": [list(item) for item in self.failure_audit_refs],
        }


def customer_strategy_id(strategy: CustomerStrategy | None) -> str:
    if strategy is None:
        return sha256_json({"mode": "native_no_overlay"})[:16]
    return sha256_json(strategy.to_dict())[:16]


def service_strategy_id(strategy: ServiceStrategy) -> str:
    return sha256_json(strategy.to_dict())[:16]
