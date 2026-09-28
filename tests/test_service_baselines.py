from __future__ import annotations

import json
from dataclasses import replace

import pytest

from evotau.records import (
    EpisodeRecord,
    EpisodeStatus,
    EvidenceRef,
    FailureRecord,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.service_baselines import InitialAttackEvidence, OneShotServiceTransition
from evotau.service_transition import GatedServiceTransition
from evotau.strategies import CustomerStrategy, ServiceRule, ServiceStrategy


def failure(*, generation: int = 0, failure_id: str = "initial-failure") -> FailureRecord:
    service = ServiceStrategy()
    customer = CustomerStrategy()

    def episode(episode_id: str, seed: int) -> EpisodeRecord:
        return EpisodeRecord(
            episode_id=episode_id,
            task_id="E-1",
            seed=seed,
            customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service),
            status=EpisodeStatus.COMPLETE,
            task_success=False,
            customer_valid=True,
            strategy_applicable=True,
            customer_strategy_adherent=True,
            policy_violation=True,
            policy_rule_id="retail.policy:explicit_confirmation",
            mistake_type="missing_explicit_confirmation",
            workflow_stage="pre_write",
            evidence=(EvidenceRef(2, "tool", "write before confirmation"),),
        )

    return FailureRecord.verify(
        episode(f"{failure_id}-source", 1),
        reproduction_episode=episode(f"{failure_id}-replay", 1001),
        generation=generation,
        verifier="review:source",
        reproduction_verifier="review:reproduction",
    )


def test_initial_attack_evidence_is_hashed_and_round_trips_strictly():
    initial = failure()
    frozen = InitialAttackEvidence.freeze(
        stage_id="preregistered-initial-attacks-v1",
        task_ids=("E-1",),
        initial_service=ServiceStrategy(),
        failures=(initial,),
    )

    restored = InitialAttackEvidence.from_dict(json.loads(json.dumps(frozen.to_dict())))

    assert restored == frozen
    assert restored.sha256 == frozen.sha256
    changed = frozen.to_dict()
    changed["failures"][0]["verification_ref"] = "changed-review"
    with pytest.raises(ValueError, match="SHA-256"):
        InitialAttackEvidence.from_dict(changed)


def gated_transition(service: ServiceStrategy) -> GatedServiceTransition:
    return GatedServiceTransition(
        evolution_task_id="E-1",
        validation_task_id="V-1",
        seed=10,
        initial_service=service,
        proposal_provider=lambda _repair_input: None,
        audit_provider=lambda _proposal, _repair_input: None,
        token_counter=lambda text: len(text.split()),
    )


def test_one_shot_transition_filters_later_failures_and_reuses_the_same_gate(monkeypatch):
    initial = failure()
    later = replace(
        failure(generation=1, failure_id="later-failure"),
        episode_id="later-source",
        reproduction_episode_id="later-replay",
    )
    evidence = InitialAttackEvidence.freeze(
        stage_id="initial-attack-collection",
        task_ids=("E-1",),
        initial_service=ServiceStrategy(),
        failures=(initial,),
    )
    forwarded = []

    def delegate(self, *args):
        assert self is transition
        forwarded.append(args)
        return args[2], None, "delegated to common paired gate"

    transition = gated_transition(evidence.initial_service)
    monkeypatch.setattr(GatedServiceTransition, "__call__", delegate)
    one_shot = OneShotServiceTransition(evidence, transition)
    customer = CustomerStrategy()
    service = evidence.initial_service
    runner = object()

    result = one_shot(0, customer, service, (later, initial), runner, None)
    later_result = one_shot(1, customer, service, (later,), runner, None)

    assert result == (service, None, "delegated to common paired gate")
    assert forwarded == [(0, customer, service, (initial,), runner, None)]
    assert later_result == (service, None, "one-shot baseline keeps the generation-0 Service fixed")
    assert one_shot.__evotau_provenance__()["initial_attack_evidence_sha256"] == evidence.sha256


def test_one_shot_transition_rejects_missing_or_changed_frozen_evidence():
    initial = failure()
    evidence = InitialAttackEvidence.freeze(
        stage_id="initial-attack-collection",
        task_ids=("E-1",),
        initial_service=ServiceStrategy(),
        failures=(initial,),
    )
    one_shot = OneShotServiceTransition(evidence, gated_transition(evidence.initial_service))

    with pytest.raises(ValueError, match="missing frozen initial evidence"):
        one_shot(0, CustomerStrategy(), evidence.initial_service, (), object(), None)
    with pytest.raises(ValueError, match="differs from frozen initial evidence"):
        one_shot(
            0,
            CustomerStrategy(),
            evidence.initial_service,
            (replace(initial, verification_ref="different-review"),),
            object(),
            None,
        )
    with pytest.raises(ValueError, match="frozen initial Service"):
        one_shot(
            0,
            CustomerStrategy(),
            ServiceStrategy((ServiceRule(
                "different", "retail.policy:other", "trigger", "execution", ("evidence",),
            ),)),
            (initial,),
            object(),
            None,
        )


def test_initial_evidence_rejects_later_generation_and_foreign_service_failures():
    with pytest.raises(ValueError, match="generation 0"):
        InitialAttackEvidence.freeze(
            stage_id="initial-attack-collection",
            task_ids=("E-1",),
            initial_service=ServiceStrategy(),
            failures=(failure(generation=1),),
        )
    with pytest.raises(ValueError, match="initial Service"):
        InitialAttackEvidence.freeze(
            stage_id="initial-attack-collection",
            task_ids=("E-1",),
            initial_service=ServiceStrategy(),
            failures=(replace(failure(), service_strategy_id="f" * 16),),
        )
