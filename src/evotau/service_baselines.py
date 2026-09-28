"""Frozen Service controls used by the RQ2 baseline protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .budget import RequestBudget
from .lifecycle import ServiceTransitionEpisodeRunner
from .manifest import sha256_json
from .records import FailureRecord, service_strategy_id
from .service_evolution import GateReport
from .service_transition import GatedServiceTransition
from .strategies import CustomerStrategy, ServiceStrategy


@dataclass(frozen=True, slots=True)
class InitialAttackEvidence:
    """Verified failures frozen from the preregistered initial attack stage."""

    stage_id: str
    task_ids: tuple[str, ...]
    initial_service: ServiceStrategy
    failures: tuple[FailureRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise ValueError("initial attack evidence requires a frozen stage ID")
        if (not isinstance(self.task_ids, tuple) or not self.task_ids
                or len(set(self.task_ids)) != len(self.task_ids)
                or any(not isinstance(item, str) or not item.strip() for item in self.task_ids)):
            raise ValueError("initial attack evidence requires ordered, unique task IDs")
        if not isinstance(self.initial_service, ServiceStrategy):
            raise TypeError("initial attack evidence requires its frozen initial Service")
        if (not isinstance(self.failures, tuple)
                or any(not isinstance(item, FailureRecord) for item in self.failures)):
            raise TypeError("initial attack evidence failures must be an immutable tuple of FailureRecords")
        failure_ids = tuple(item.failure_id for item in self.failures)
        if len(set(failure_ids)) != len(failure_ids):
            raise ValueError("initial attack evidence cannot repeat a failure")
        if failure_ids != tuple(sorted(failure_ids)):
            raise ValueError("initial attack failures must be sorted by failure ID")
        initial_service_id = service_strategy_id(self.initial_service)
        for failure in self.failures:
            if failure.generation != 0:
                raise ValueError("one-shot repair evidence must come from generation 0")
            if failure.task_id not in self.task_ids:
                raise ValueError("one-shot repair evidence contains a task outside its frozen panel")
            if failure.service_strategy_id != initial_service_id:
                raise ValueError("one-shot repair evidence must come from the initial Service")

    @classmethod
    def freeze(
        cls,
        *,
        stage_id: str,
        task_ids: tuple[str, ...],
        initial_service: ServiceStrategy,
        failures: tuple[FailureRecord, ...],
    ) -> InitialAttackEvidence:
        """Create a deterministically ordered immutable snapshot of initial evidence."""

        if not isinstance(failures, tuple) or any(
            not isinstance(item, FailureRecord) for item in failures
        ):
            raise TypeError("initial attack failures must be an immutable tuple of FailureRecords")
        return cls(
            stage_id=stage_id,
            task_ids=task_ids,
            initial_service=initial_service,
            failures=tuple(sorted(failures, key=lambda item: item.failure_id)),
        )

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": 1,
            "stage_id": self.stage_id,
            "task_ids": list(self.task_ids),
            "initial_service": self.initial_service.to_dict(),
            "initial_service_strategy_id": service_strategy_id(self.initial_service),
            "failures": [item.to_dict() for item in self.failures],
        }
        return {**body, "sha256": sha256_json(body)}

    @classmethod
    def from_dict(cls, value: Any) -> InitialAttackEvidence:
        fields = {
            "schema_version", "stage_id", "task_ids", "initial_service",
            "initial_service_strategy_id", "failures", "sha256",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("initial attack evidence has missing or unknown fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported initial attack evidence schema_version")
        if not isinstance(value["task_ids"], list) or not isinstance(value["failures"], list):
            raise TypeError("initial attack task IDs and failures must be JSON arrays")
        service_value = value["initial_service"]
        if not isinstance(service_value, dict) or set(service_value) != {"rules"}:
            raise ValueError("initial Service strategy is malformed")
        if not isinstance(service_value["rules"], list):
            raise TypeError("initial Service rules must be a JSON array")
        from .strategies import ServiceRule

        for rule in service_value["rules"]:
            if not isinstance(rule, dict) or set(rule) != {
                "rule_id", "policy_ref", "trigger", "required_execution", "evidence_refs",
            }:
                raise ValueError("initial Service rule is malformed")
            if (not isinstance(rule["evidence_refs"], list)
                    or any(not isinstance(ref, str) for ref in rule["evidence_refs"])):
                raise ValueError("initial Service rule evidence_refs must be strings")
        if any(not isinstance(item, dict) for item in value["failures"]):
            raise ValueError("initial attack failure is malformed")
        try:
            initial_service = ServiceStrategy(tuple(
                ServiceRule(
                    rule["rule_id"], rule["policy_ref"], rule["trigger"],
                    rule["required_execution"], tuple(rule["evidence_refs"]),
                )
                for rule in service_value["rules"]
            ))
            result = cls(
                stage_id=value["stage_id"],
                task_ids=tuple(value["task_ids"]),
                initial_service=initial_service,
                failures=tuple(FailureRecord.from_dict(item) for item in value["failures"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"invalid initial attack evidence: {exc}") from exc
        if value["initial_service_strategy_id"] != service_strategy_id(initial_service):
            raise ValueError("initial attack evidence Service strategy ID does not match")
        if value["sha256"] != result.sha256:
            raise ValueError("initial attack evidence SHA-256 does not match its contents")
        return result

    @property
    def sha256(self) -> str:
        return self.to_dict()["sha256"]

    @property
    def failure_ids(self) -> tuple[str, ...]:
        return tuple(item.failure_id for item in self.failures)


@dataclass(frozen=True, slots=True)
class OneShotServiceTransition:
    """Run the normal gated repair once, using only frozen generation-0 evidence.

    Supply this callback to ``TwoGenerationSmoke`` as its Service transition.
    It delegates the first-generation proposal, independent audit, and paired
    gate to the same ``GatedServiceTransition`` used by adaptive EvoTau. The
    second generation keeps the resulting Service fixed.
    """

    initial_evidence: InitialAttackEvidence
    gated_transition: GatedServiceTransition

    def __post_init__(self) -> None:
        if not isinstance(self.gated_transition, GatedServiceTransition):
            raise TypeError("one-shot Service baseline requires the native GatedServiceTransition")

    def __call__(
        self,
        generation: int,
        customer: CustomerStrategy,
        incumbent: ServiceStrategy,
        failures: tuple[FailureRecord, ...],
        episode_runner: ServiceTransitionEpisodeRunner,
        request_budget: RequestBudget | None,
    ) -> tuple[ServiceStrategy, GateReport | None, str]:
        if generation not in (0, 1):
            raise ValueError("one-shot Service baseline supports generations 0 and 1")
        if generation == 1:
            return incumbent, None, "one-shot baseline keeps the generation-0 Service fixed"
        if service_strategy_id(incumbent) != service_strategy_id(self.initial_evidence.initial_service):
            raise ValueError("one-shot Service baseline must start from its frozen initial Service")

        supplied = {item.failure_id: item for item in failures}
        expected = {item.failure_id: item for item in self.initial_evidence.failures}
        if len(supplied) != len(failures):
            raise ValueError("one-shot Service failure pool contains duplicate failure IDs")
        for failure_id, frozen in expected.items():
            observed = supplied.get(failure_id)
            if observed is None:
                raise ValueError("one-shot Service failure pool is missing frozen initial evidence")
            if observed != frozen:
                raise ValueError("one-shot Service failure differs from frozen initial evidence")
        return self.gated_transition(
            generation,
            customer,
            incumbent,
            self.initial_evidence.failures,
            episode_runner,
            request_budget,
        )

    def __evotau_provenance__(self) -> dict[str, Any]:
        """Expose non-secret immutable control inputs for native resume identity."""

        return {
            "schema_version": 1,
            "baseline": "one_shot_repair",
            "initial_attack_evidence_sha256": self.initial_evidence.sha256,
            "initial_attack_stage_id": self.initial_evidence.stage_id,
            "task_ids": list(self.initial_evidence.task_ids),
            "initial_service_strategy_id": service_strategy_id(
                self.initial_evidence.initial_service
            ),
            "eligible_failure_ids": list(self.initial_evidence.failure_ids),
        }
