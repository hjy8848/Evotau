from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from threading import Barrier, Lock
from time import sleep
from types import SimpleNamespace

import pytest

from evotau.archive import FailureArchive
from evotau.attribution import confirm_failure_reproductions, promote_verified_failure
from evotau.budget import BudgetSnapshot, RequestBudget
from evotau.checkpoint import (
    EvolutionCheckpoint,
    load_checkpoint,
    manifest_fingerprint,
    save_checkpoint,
)
from evotau.crossplay import build_crossplay_matrix
from evotau.lifecycle import (
    TwoGenerationSmoke,
    evaluate_customer_panel,
    run_customer_round,
)
from evotau.manifest import MVP_FAILURE_TAXONOMY, MechanismManifest, sha256_json
from evotau.mutation import propose_customer_candidates
from evotau.phase0 import load_config
from evotau.records import (
    CandidateEvaluation,
    EpisodeRecord,
    EpisodeStatus,
    EvidenceRef,
    FailureRecord,
    FailureSignature,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.selection import select_customer
from evotau.service_evolution import (
    GateReport,
    GateUnit,
    RepairAudit,
    RepairProposal,
    build_repair_candidate,
    evaluate_repair_gate,
)
from evotau.strategies import CustomerStrategy, ServiceRule, ServiceStrategy


def episode(*, name="e1", task="task-1", seed=1, customer="c", service="s",
            success=False, violation=True, status=EpisodeStatus.COMPLETE, calls=4):
    return EpisodeRecord(
        episode_id=name, task_id=task, seed=seed, customer_strategy_id=customer,
        service_strategy_id=service, status=status, task_success=success,
        customer_valid=True, strategy_applicable=True, customer_strategy_adherent=True,
        policy_violation=violation, invalid_repeated_write_calls=0,
        policy_rule_id="retail.policy:explicit_confirmation",
        mistake_type="missing_explicit_confirmation", workflow_stage="pre_write",
        evidence=(EvidenceRef(2, "tool", "write occurred before explicit confirmation"),), tool_calls=calls,
    )


def replay_of(ep):
    return replace(ep, episode_id=f"{ep.episode_id}:replay", seed=ep.seed + 10_000)


def verified(ep, generation=0, replay=None, verifier=None, reproduction_verifier=None):
    replay = replay or replay_of(ep)
    return FailureRecord.verify(
        ep, reproduction_episode=replay, generation=generation,
        verifier=verifier or f"audit:source:{ep.episode_id}",
        reproduction_verifier=reproduction_verifier or f"audit:replay:{replay.episode_id}",
    )


def paired_evaluation(episode_record, replay, *, panel="discovery"):
    failure = verified(episode_record, replay=replay)
    refs = (
        (episode_record.episode_id, failure.verification_ref),
        (replay.episode_id, failure.reproduction_verification_ref),
    )
    return CandidateEvaluation(
        episode_record.customer_strategy_id, (episode_record,), (failure,), panel,
        refs, (replay,),
    )


def test_attribution_requires_validity_adherence_evidence_and_independent_audit():
    ep = episode()
    replay = replay_of(ep)
    assert not promote_verified_failure(ep, generation=0, independent_verification_ref=None).promoted
    assert not promote_verified_failure(
        ep, generation=0, independent_verification_ref="audit:1",
    ).promoted
    assert promote_verified_failure(
        ep, generation=0, independent_verification_ref="audit:1",
        reproduction_episode=replay, reproduction_verification_ref="audit:replay:1",
    ).promoted
    assert not promote_verified_failure(replace(ep, customer_valid=False), generation=0,
                                        independent_verification_ref="audit:1").promoted
    assert not promote_verified_failure(replace(ep, evidence=()), generation=0,
                                        independent_verification_ref="audit:1").promoted
    out_of_scope = replace(
        ep, policy_rule_id="retail.policy:refund_limit", mistake_type="unauthorized_refund",
        workflow_stage="decision",
    )
    decision = promote_verified_failure(
        out_of_scope, generation=0, independent_verification_ref="audit:out-of-scope",
        reproduction_episode=replay_of(out_of_scope),
        reproduction_verification_ref="audit:out-of-scope-replay",
    )
    assert not decision.promoted
    assert "outside the frozen MVP taxonomy" in decision.reason
    with pytest.raises(ValueError, match="outside the frozen Retail MVP taxonomy"):
        FailureRecord.verify(
            out_of_scope, reproduction_episode=replay_of(out_of_scope), generation=0,
            verifier="audit:out-of-scope", reproduction_verifier="audit:out-of-scope-replay",
        )
    forged_policy_ref = replace(ep, policy_rule_id="retail.policy:refund_limit")
    assert not promote_verified_failure(
        forged_policy_ref, generation=0, independent_verification_ref="audit:wrong-policy-ref",
        reproduction_episode=replay_of(forged_policy_ref),
        reproduction_verification_ref="audit:wrong-policy-replay",
    ).promoted


def test_failure_promotion_requires_audited_fresh_seed_strategy_reproduction():
    source = episode(name="source", seed=17)
    same_strategy_replay = replace(source, episode_id="replay", seed=117)
    accepted = promote_verified_failure(
        source, generation=0, independent_verification_ref="audit:source",
        reproduction_episode=same_strategy_replay,
        reproduction_verification_ref="audit:replay",
    )
    assert accepted.promoted
    assert accepted.failure is not None
    assert accepted.failure.episode_id == "source"
    assert accepted.failure.reproduction_episode_id == "replay"
    assert accepted.failure.reproduction_verification_ref == "audit:replay"

    same_seed = promote_verified_failure(
        source, generation=0, independent_verification_ref="audit:source",
        reproduction_episode=replace(same_strategy_replay, seed=source.seed),
        reproduction_verification_ref="audit:same-seed",
    )
    assert not same_seed.promoted and "fresh seed" in same_seed.reason
    mismatched_signature = replace(
        same_strategy_replay, workflow_stage="identity_verification",
        policy_rule_id="retail.policy:identity_verification",
        mistake_type="missing_identity_verification",
    )
    mismatched = promote_verified_failure(
        source, generation=0, independent_verification_ref="audit:source",
        reproduction_episode=mismatched_signature,
        reproduction_verification_ref="audit:different-signature",
    )
    assert not mismatched.promoted and "exact failure signature" in mismatched.reason


def test_confirmation_panel_reproduces_only_exact_adherent_failure_signatures():
    source = episode(name="discovery", task="task-1", seed=3, customer="same-c", service="same-s")
    replay = replace(source, episode_id="confirmation", seed=103)
    discovery = CandidateEvaluation(
        "same-c", (source,), panel_name="discovery",
        failure_audit_refs=((source.episode_id, "audit:discovery"),),
    )
    confirmation = CandidateEvaluation(
        "same-c", (replay,), panel_name="confirmation",
        failure_audit_refs=((replay.episode_id, "audit:confirmation"),),
    )
    confirmed_source, confirmed_replay, failures = confirm_failure_reproductions(
        discovery, confirmation, generation=2,
    )
    assert len(failures) == 1
    assert confirmed_source.fitness == 1
    assert confirmed_replay.fitness == 0
    assert confirmed_source.verified_failures[0].reproduction_episode_id == "confirmation"
    assert not confirmed_replay.verified_failures

    different = replace(
        replay, policy_rule_id="retail.policy:identity_verification",
        workflow_stage="identity_verification", mistake_type="missing_identity_verification",
    )
    different_panel = CandidateEvaluation(
        "same-c", (different,), panel_name="confirmation",
        failure_audit_refs=((different.episode_id, "audit:different"),),
    )
    source_unmatched, replay_unmatched, no_failures = confirm_failure_reproductions(
        discovery, different_panel, generation=2,
    )
    assert not no_failures
    assert source_unmatched.fitness == replay_unmatched.fitness == 0


def test_tied_customer_fitness_can_archive_a_reproduced_new_signature_without_replacement():
    incumbent = CustomerStrategy()
    service = ServiceStrategy()
    candidate = propose_customer_candidates(incumbent, 2, seed=41)[0]
    customer_id = customer_strategy_id(incumbent)
    candidate_id = candidate.strategy_id
    service_id = service_strategy_id(service)

    def runner(*, task_id, seed, customer, service, panel_name):
        strategy_id = customer_strategy_id(customer)
        is_incumbent = strategy_id == customer_id
        is_candidate = strategy_id == candidate_id
        violation = is_incumbent or is_candidate
        if is_incumbent:
            policy, stage, mistake = (
                "retail.policy:explicit_confirmation", "pre_write", "missing_explicit_confirmation",
            )
        else:
            policy, stage, mistake = (
                "retail.policy:identity_verification", "identity_verification",
                "missing_identity_verification",
            )
        return EpisodeRecord(
            episode_id=f"{panel_name}:{strategy_id}:{seed}",
            task_id=task_id,
            seed=seed,
            customer_strategy_id=strategy_id,
            service_strategy_id=service_id,
            status=EpisodeStatus.COMPLETE,
            task_success=not violation,
            customer_valid=True,
            strategy_applicable=True,
            customer_strategy_adherent=True,
            policy_violation=violation,
            policy_rule_id=policy if violation else None,
            workflow_stage=stage if violation else None,
            mistake_type=mistake if violation else None,
            evidence=(EvidenceRef(2, "tool", "policy step missing"),) if violation else (),
        )

    result = run_customer_round(
        runner, incumbent=incumbent, service=service, task_ids=("E",), seeds=(7,),
        generation=0, proposal_seed=41, count=2,
        failure_verifier=lambda item: f"review:{item.episode_id}",
        confirmation_task_ids=("E",), confirmation_seeds=(10007,),
    )

    assert not result.selection.evolved
    assert result.selection.discovery_scores[0][1] == 1
    candidate_eval = next(item for item in result.candidates if item.strategy_id == candidate_id)
    assert candidate_eval.fitness == 1
    assert candidate_eval.signature_count == 1
    assert {failure.signature.key for failure in result.verified_failures} == {
        candidate_eval.verified_failures[0].signature.key,
        result.incumbent.verified_failures[0].signature.key,
    }
    assert len(result.verified_failures) == 2  # confirmation rows are evidence, not duplicate discoveries


def test_customer_failure_confirmation_schedule_is_preflighted_before_dispatch():
    dispatches = 0

    def runner(**_kwargs):
        nonlocal dispatches
        dispatches += 1
        raise AssertionError("invalid confirmation schedule must fail before an episode runs")

    with pytest.raises(ValueError, match="fresh task/seed pairs"):
        run_customer_round(
            runner, incumbent=CustomerStrategy(), service=ServiceStrategy(),
            task_ids=("E",), seeds=(13,), generation=0, proposal_seed=13,
            confirmation_task_ids=("E",), confirmation_seeds=(13,),
        )
    assert dispatches == 0


def test_customer_mutation_determinism_and_incumbent_retention_on_tie():
    incumbent = CustomerStrategy()
    a = propose_customer_candidates(incumbent, 2, seed=17)
    b = propose_customer_candidates(incumbent, 2, seed=17)
    assert [x.strategy_id for x in a] == [x.strategy_id for x in b]
    assert len({x.strategy_id for x in a}) == 2
    all_axes = propose_customer_candidates(incumbent, 3, seed=17)
    assert {field for candidate in all_axes for field in candidate.changed_fields} == {
        "disclosure", "request_order", "challenge_style", "challenge_budget",
    }
    assert all(candidate.expected_behavioral_effect for candidate in all_axes)
    incumbent_episode = episode(customer="inc", success=False)
    candidate_episode = episode(name="e2", customer="candidate", success=False)
    base = paired_evaluation(incumbent_episode, replay_of(incumbent_episode))
    equal = paired_evaluation(candidate_episode, replay_of(candidate_episode))
    decision = select_customer(base, (equal,))
    assert not decision.evolved and decision.selected_id == "inc"


def test_failure_conditioned_mutation_records_exact_supporting_failure_lineage():
    failure = verified(episode())
    proposals = propose_customer_candidates(CustomerStrategy(), 3, seed=13, recent_failures=(failure,))
    by_operator = {proposal.operator: proposal for proposal in proposals}
    assert by_operator["request_order"].rationale == "failure_conditioned"
    assert by_operator["request_order"].supporting_failure_ids == (failure.failure_id,)
    assert by_operator["challenge"].supporting_failure_ids == (failure.failure_id,)
    assert by_operator["challenge"].rationale == "failure_conditioned"
    assert by_operator["disclosure"].supporting_failure_ids == ()


def test_strict_customer_win_requires_fresh_paired_confirmation():
    base_1 = episode(name="b1", task="task-1", customer="incumbent")
    base_2 = episode(name="b2", task="task-2", seed=2, customer="incumbent")
    candidate_1 = episode(name="c1", task="task-1", customer="challenger")
    candidate_2 = episode(name="c2", task="task-2", seed=2, customer="challenger")
    old_v1 = replace(base_1, episode_id="old-v1", seed=101)
    old_v2 = replace(base_2, episode_id="old-v2", seed=102)
    new_v1 = replace(candidate_1, episode_id="new-v1", seed=101)
    new_v2 = replace(candidate_2, episode_id="v2-new", seed=202)
    incumbent = CandidateEvaluation(
        "incumbent", (base_1, base_2),
        (verified(base_1, replay=old_v1),),
        failure_audit_refs=(
            (base_1.episode_id, "audit:source:b1"), (old_v1.episode_id, "audit:replay:old-v1"),
        ),
        replication_episodes=(old_v1,),
    )
    challenger = CandidateEvaluation(
        "challenger", (candidate_1, candidate_2),
        (verified(candidate_1, replay=new_v1), verified(candidate_2, replay=new_v2)),
        failure_audit_refs=(
            (candidate_1.episode_id, "audit:source:c1"), (new_v1.episode_id, "audit:replay:new-v1"),
            (candidate_2.episode_id, "audit:source:c2"), (new_v2.episode_id, "audit:replay:v2-new"),
        ),
        replication_episodes=(new_v1, new_v2),
    )
    no_confirmation = select_customer(incumbent, (challenger,))
    assert not no_confirmation.evolved
    old_v2 = replace(base_2, episode_id="v2-old", seed=202, task_success=True,
                     policy_violation=False, evidence=())
    old_v = CandidateEvaluation(
        "incumbent", (old_v1, old_v2), panel_name="confirmation",
        failure_audit_refs=((old_v1.episode_id, "audit:replay:old-v1"),),
    )
    new_v = CandidateEvaluation(
        "challenger", (new_v1, new_v2), panel_name="confirmation",
        failure_audit_refs=(
            (new_v1.episode_id, "audit:replay:new-v1"),
            (new_v2.episode_id, "audit:replay:v2-new"),
        ),
    )
    confirmed = select_customer(incumbent, (challenger,),
                                confirmation={"incumbent": old_v, "challenger": new_v})
    assert confirmed.evolved and confirmed.selected_id == "challenger"


def test_customer_fitness_ignores_invalid_and_unverified_episodes():
    strategy, service = CustomerStrategy(), ServiceStrategy()
    ids = customer_strategy_id(strategy), service_strategy_id(service)

    def runner(**kwargs):
        return episode(name=f"{kwargs['task_id']}-{kwargs['seed']}", task=kwargs["task_id"],
                       seed=kwargs["seed"], customer=ids[0], service=ids[1])

    result = evaluate_customer_panel(
        runner, task_ids=("a",), seeds=(1,), strategy=strategy, service=service,
        panel_name="discovery", generation=0,
    )
    assert result.fitness == 0  # no independent verification artifact
    verified_result = evaluate_customer_panel(
        runner, task_ids=("a",), seeds=(1,), strategy=strategy, service=service,
        panel_name="discovery", generation=0,
        strategy_seen_failures={"a-1": "audit:fixture"},
    )
    assert verified_result.fitness == 0  # one audited trace is still only a provisional signal
    assert verified_result.provisional_failure_count == 1


def test_strategy_not_applicable_is_separate_from_nonadherence_and_customer_invalidity():
    customer = CustomerStrategy()
    applicable = episode(customer=customer_strategy_id(customer))
    nonadherent = replace(
        applicable, episode_id="not-adherent", status=EpisodeStatus.INVALID_STRATEGY,
        task_success=False, customer_strategy_adherent=False, policy_violation=False, evidence=(),
    )
    not_applicable = replace(
        applicable, episode_id="not-applicable", strategy_applicable=False,
        customer_strategy_adherent=None, policy_violation=False, evidence=(),
    )
    evaluation = CandidateEvaluation(
        customer_strategy_id(customer), (applicable, nonadherent, not_applicable),
    )
    assert evaluation.valid_episode_count == 3
    assert evaluation.invalid_episode_count == 0
    assert evaluation.uncertain_episode_count == 0
    assert evaluation.strategy_opportunity_count == 2
    assert evaluation.strategy_adherent_count == 1
    assert evaluation.strategy_not_applicable_count == 1
    assert evaluation.strategy_adherence_rate == 0.5


def test_candidate_evaluation_binds_verified_failures_to_its_exact_episode():
    item = episode(customer="actual")
    failure = verified(item)
    with pytest.raises(ValueError, match="strategy does not match"):
        CandidateEvaluation("other-customer", (item,), (failure,))
    with pytest.raises(ValueError, match="refer to an episode"):
        CandidateEvaluation("actual", (replace(item, episode_id="different-episode"),), (failure,))


def test_crossplay_matrix_is_balanced_and_only_counts_verified_failures():
    customers = (CustomerStrategy(), CustomerStrategy(challenge_style="ask_reason", challenge_budget=1))
    services = (ServiceStrategy(),)
    customer_ids = tuple(customer_strategy_id(item) for item in customers)
    service_id = service_strategy_id(services[0])
    items = []
    for customer_id in customer_ids:
        for task_id in ("task-1", "task-2"):
            for seed in (1, 2):
                is_first = customer_id == customer_ids[0]
                item = episode(
                    name=f"{customer_id}:{task_id}:{seed}", task=task_id, seed=seed,
                    customer=customer_id, service=service_id,
                    success=not (is_first and task_id == "task-1" and seed == 1),
                    violation=is_first and task_id == "task-1" and seed == 1,
                )
                if is_first and task_id == "task-2" and seed == 1:
                    item = replace(item, status=EpisodeStatus.INVALID_CUSTOMER,
                                   customer_valid=False, customer_strategy_adherent=False,
                                   task_success=False, policy_violation=False, evidence=())
                elif is_first and task_id == "task-2" and seed == 2:
                    item = replace(item, status=EpisodeStatus.INFRASTRUCTURE_ERROR,
                                   task_success=None, policy_violation=False, evidence=())
                elif not is_first and task_id == "task-1" and seed == 1:
                    # Native review flagged this trajectory, but no independent
                    # verifier promoted it into FailureRecord evidence.
                    item = replace(item, policy_violation=True)
                items.append(item)
    verified_failure = verified(items[0])
    matrix = build_crossplay_matrix(
        items,
        (verified_failure,),
        customer_strategies=customers,
        service_strategies=services,
        task_ids=("task-1", "task-2"),
        seeds=(1, 2),
    )
    first, second = matrix.cells
    assert (first.attempted_episodes, first.valid_episodes, first.invalid_episodes,
            first.infrastructure_episodes, first.uncertain_episodes) == (4, 2, 1, 1, 0)
    assert first.verified_failure_rate == 0.5
    assert first.strategy_opportunities == 2
    assert first.strategy_adherent_episodes == 2
    assert first.strategy_not_applicable_episodes == 0
    assert first.strategy_adherence_rate == 1.0
    assert first.unique_task_signature_failures == 1
    assert first.unique_signatures == 1
    assert second.verified_failure_rate == 0.0
    assert second.verified_failure_episodes == 0
    assert second.unique_signatures == 0


def test_crossplay_rejects_unbalanced_panels_and_unlinked_verifications():
    customer, service = CustomerStrategy(), ServiceStrategy()
    customer_id, service_id = customer_strategy_id(customer), service_strategy_id(service)
    item = episode(customer=customer_id, service=service_id)
    kwargs = {
        "customer_strategies": (customer,), "service_strategies": (service,),
        "task_ids": (item.task_id, "task-2"), "seeds": (item.seed,),
    }
    with pytest.raises(ValueError, match="missing task/seed"):
        build_crossplay_matrix((item,), (), **kwargs)
    unrelated = verified(replace(item, episode_id="not-in-matrix"))
    full_panel = (item, replace(item, episode_id="second", task_id="task-2"))
    with pytest.raises(ValueError, match="outside the cross-play"):
        build_crossplay_matrix(full_panel, (unrelated,), **kwargs)


def test_crossplay_separates_native_clean_customer_from_not_applicable_attack():
    clean_id = customer_strategy_id(None)
    adversary = CustomerStrategy()
    service = ServiceStrategy()
    clean = episode(
        name="clean", customer=clean_id, service=service_strategy_id(service), violation=False,
    )
    clean = replace(clean, strategy_applicable=False, customer_strategy_adherent=None)
    attack = episode(
        name="attack", customer=customer_strategy_id(adversary),
        service=service_strategy_id(service), violation=False,
    )
    attack = replace(attack, invalid_repeated_write_calls=None)
    matrix = build_crossplay_matrix(
        (clean, attack), (), customer_strategies=(None, adversary), service_strategies=(service,),
        task_ids=(clean.task_id,), seeds=(clean.seed,),
    )
    clean_cell = next(item for item in matrix.cells if item.customer_strategy_id == clean_id)
    assert clean_cell.valid_episodes == 1
    assert clean_cell.strategy_opportunities == 0
    assert clean_cell.strategy_not_applicable_episodes == 1
    assert clean_cell.strategy_adherence_rate is None
    assert clean_cell.repeated_write_audit_coverage == 1.0
    attack_cell = next(item for item in matrix.cells if item.customer_strategy_id == customer_strategy_id(adversary))
    assert attack_cell.repeated_write_audited_episodes == 0
    assert attack_cell.repeated_write_audit_coverage == 0.0


def test_archive_is_append_only_idempotent_and_deduplicates_representatives(tmp_path):
    archive = FailureArchive(tmp_path / "failures.sqlite")
    customer = CustomerStrategy()
    customer_id = customer_strategy_id(customer)
    assert archive.append_customer_strategy(
        customer, parent_id=None, operator="incumbent", generation=0,
    )
    assert archive.get_customer_strategy(customer_id) == customer
    assert archive.get_customer_strategy("missing-customer") is None
    with pytest.raises(ValueError, match="non-empty strategy ID"):
        archive.get_customer_strategy(" ")

    one = verified(episode(name="one"), 0)
    two = replace(verified(episode(name="two"), 1), signature=one.signature)
    assert archive.append(one)
    assert not archive.append(one)
    assert archive.append(two)
    assert archive.recurrence_count(one.signature.key) == 2
    assert len(archive.recent()) == 2
    assert len(archive.representatives()) == 1
    assert len(archive.active_representatives()) == 1
    assert archive.active_replay_coverage() == {
        "active_signatures": 1, "replayed_signatures": 0,
        "uncovered_signatures": 1, "coverage_rate": 0.0,
    }
    active = archive.active_representatives()[0]
    assert archive.mark_replayed(active.failure_id, generation=2)
    assert not archive.mark_replayed(active.failure_id, generation=2)
    assert archive.active_replay_coverage()["coverage_rate"] == 1.0


def test_archive_keeps_legacy_single_episode_rows_but_never_reuses_them(tmp_path):
    path = tmp_path / "legacy-failures.sqlite"
    archive = FailureArchive(path)
    legacy = verified(episode(name="legacy"), 0)
    assert archive.append(legacy)

    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT payload FROM failures WHERE failure_id=?", (legacy.failure_id,),
        ).fetchone()
        payload = json.loads(row[0])
        payload.pop("reproduction_episode_id")
        payload.pop("reproduction_verification_ref")
        db.execute(
            "UPDATE failures SET payload=? WHERE failure_id=?",
            (json.dumps(payload, sort_keys=True), legacy.failure_id),
        )
        db.execute(
            "UPDATE failures SET protocol_version=1 WHERE failure_id=?", (legacy.failure_id,),
        )

    assert archive.recent() == ()
    assert archive.representatives() == ()
    assert archive.active_representatives() == ()
    assert archive.recurrence_count(legacy.signature.key) == 0
    with pytest.raises(ValueError, match="unreproduced"):
        archive.mark_replayed(legacy.failure_id, generation=1)

    reproduced = replace(legacy, failure_id="new-reproduced-row")
    assert archive.append(reproduced)
    assert archive.recent() == (reproduced,)
    assert archive.recurrence_count(legacy.signature.key) == 1


def test_archive_active_representatives_obey_cap_and_prioritize_recent_recurrence(tmp_path):
    archive = FailureArchive(tmp_path / "bounded-failures.sqlite")
    base = verified(episode(name="base"), 0)
    signatures = []
    # The frozen MVP taxonomy contains only three exact signature classes, so
    # the 32-item limit cannot be saturated without inventing out-of-scope data.
    for index, (stage, policy_ref, mistake) in enumerate(MVP_FAILURE_TAXONOMY):
        signature = FailureSignature("retail", stage, policy_ref, mistake)
        item = replace(
            base,
            failure_id=f"failure-{index:02d}",
            episode_id=f"episode-{index:02d}",
            task_id=f"task-{index:02d}",
            generation=1,
            signature=signature,
            policy_ref=signature.policy_rule_id,
            severity="high" if index == 2 else "material",
        )
        archive.append(item)
        signatures.append(item)
    recurring = replace(
        signatures[0], failure_id="failure-recurrence", episode_id="episode-recurrence",
        generation=2,
    )
    archive.append(recurring)

    active = archive.active_representatives(current_generation=2)

    assert len(active) == len(MVP_FAILURE_TAXONOMY)
    assert active[0].failure_id == "failure-recurrence"
    assert active[1].failure_id == "failure-02"
    assert len({item.signature.key for item in active}) == len(MVP_FAILURE_TAXONOMY)
    assert archive.active_replay_coverage()["active_signatures"] == len(MVP_FAILURE_TAXONOMY)
    assert len(archive.active_representatives(limit=2, current_generation=2)) == 2
    with pytest.raises(ValueError, match="capped"):
        archive.active_representatives(33)


def test_service_repair_audit_and_paired_gate():
    failure = verified(episode())
    incumbent = ServiceStrategy()
    rule = ServiceRule("r1", failure.policy_ref, "before a database write",
                       "summarize all action details and get explicit yes confirmation before writing",
                       failure.evidence_refs)
    audit = RepairAudit(
        True, "audit:policy", frozenset({failure.policy_ref}),
        rationale="The rule reinforces the fixed confirmation requirement without adding permissions.",
    )
    with pytest.raises(ValueError, match="falsifiable verification hypothesis"):
        RepairProposal(failure.failure_id, rule, " ")
    proposal = RepairProposal(
        failure.failure_id, rule,
        "On the same eligible write request, the confirmation is checked before the write tool call.",
    )
    candidate = build_repair_candidate(incumbent, proposal, failure,
                                       audit, token_counter=lambda text: len(text.split()))
    old_service_id = service_strategy_id(incumbent)
    candidate_service_id = service_strategy_id(candidate)
    old_target = episode(name="old-target", service=old_service_id)
    new_target = replace(old_target, episode_id="new-target", service_strategy_id=candidate_service_id,
                         task_success=True, policy_violation=False)
    old_clean = replace(old_target, episode_id="old-clean", task_id="clean", task_success=True,
                        strategy_applicable=False, customer_strategy_adherent=None,
                        policy_violation=False, tool_calls=3)
    new_clean = replace(old_clean, episode_id="new-clean", service_strategy_id=candidate_service_id,
                        tool_calls=4)
    initial_s0_clean = replace(old_clean, episode_id="initial-s0-clean")
    old_valid = replace(old_target, episode_id="old-valid", task_id="validation", task_success=True,
                        policy_violation=False)
    new_valid = replace(old_valid, episode_id="new-valid", service_strategy_id=candidate_service_id)
    old_target_2 = replace(old_target, episode_id="old-target-2", seed=2)
    new_target_2 = replace(new_target, episode_id="new-target-2", seed=2)
    historical_old = replace(old_target, episode_id="old-history", task_id="history", task_success=True,
                             policy_violation=False)
    historical_new = replace(historical_old, episode_id="new-history",
                             service_strategy_id=candidate_service_id)
    units = (
        GateUnit("target-1", "target", old_target, new_target, failure.failure_id),
        GateUnit("target-2", "target", old_target_2, new_target_2, failure.failure_id),
        GateUnit("clean", "clean", old_clean, new_clean, initial_s0=initial_s0_clean),
        GateUnit("validation", "validation", old_valid, new_valid),
    )
    units += (GateUnit("history", "historical", historical_old, historical_new),)
    report = evaluate_repair_gate(
        incumbent, candidate, units, target_failure=failure, proposal=proposal, audit=audit,
        initial_service_strategy_id=old_service_id,
        token_counter=lambda text: len(text.split()),
    )
    assert report.accepted
    serialized_report = report.to_dict()
    assert serialized_report["evaluated_candidate"] == candidate.to_dict()
    assert serialized_report["proposal"]["verification_hypothesis"] == proposal.verification_hypothesis
    assert serialized_report["audit"]["verifier_ref"] == audit.verifier_ref
    assert serialized_report["initial_service_strategy_id"] == old_service_id
    assert serialized_report["initial_s0_episode_refs"] == [["clean", "initial-s0-clean"]]
    bad_audit = replace(audit, permission_delta=True)
    with pytest.raises(ValueError, match="permissions"):
        build_repair_candidate(ServiceStrategy(), proposal, failure,
                               bad_audit, token_counter=lambda text: len(text.split()))
    regressed = replace(new_clean, task_success=False)
    rejected = evaluate_repair_gate(ServiceStrategy(), candidate,
        units[:2] + (GateUnit("clean", "clean", old_clean, regressed,
                              initial_s0=initial_s0_clean),) + units[3:],
        target_failure=failure, proposal=proposal, audit=audit,
        initial_service_strategy_id=old_service_id,
        token_counter=lambda text: len(text.split()))
    assert not rejected.accepted and any("regresses" in reason for reason in rejected.reasons)
    assert rejected.candidate_strategy is None
    assert rejected.evaluated_candidate == candidate
    misbound = evaluate_repair_gate(
        incumbent,
        candidate,
        (GateUnit("target-1", "target", old_target, replace(new_target, service_strategy_id="wrong"),
                  failure.failure_id),
         *units[1:]),
        target_failure=failure, proposal=proposal, audit=audit,
        initial_service_strategy_id=old_service_id,
        token_counter=lambda text: len(text.split()),
    )
    assert not misbound.accepted and any("do not match" in reason for reason in misbound.reasons)
    wrong_target_signature = replace(
        old_target,
        policy_rule_id="retail.policy:identity_verification",
        mistake_type="missing_identity_verification",
    )
    mismatched_target = evaluate_repair_gate(
        incumbent,
        candidate,
        (GateUnit("target-1", "target", wrong_target_signature, new_target, failure.failure_id),
         *units[1:]),
        target_failure=failure,
        proposal=proposal,
        audit=audit,
        initial_service_strategy_id=old_service_id,
        token_counter=lambda text: len(text.split()),
    )
    assert not mismatched_target.accepted
    assert any("verified target failure signature" in reason for reason in mismatched_target.reasons)
    inapplicable_attack = replace(
        old_target, strategy_applicable=False, customer_strategy_adherent=None,
    )
    rejected_not_applicable = evaluate_repair_gate(
        incumbent,
        candidate,
        (GateUnit("target-1", "target", inapplicable_attack, new_target, failure.failure_id),
         *units[1:]),
        target_failure=failure, proposal=proposal, audit=audit,
        initial_service_strategy_id=old_service_id,
        token_counter=lambda text: len(text.split()),
    )
    assert not rejected_not_applicable.accepted
    assert any("applicable and adherent" in reason for reason in rejected_not_applicable.reasons)

    repeated_write_regression = evaluate_repair_gate(
        incumbent,
        candidate,
        (replace(units[0], candidate=replace(new_target, invalid_repeated_write_calls=1)),
         *units[1:]),
        target_failure=failure,
        proposal=proposal,
        audit=audit,
        initial_service_strategy_id=old_service_id,
        token_counter=lambda text: len(text.split()),
    )
    assert not repeated_write_regression.accepted
    assert any("invalid repeated write calls" in reason for reason in repeated_write_regression.reasons)

    missing_repeat_audit = evaluate_repair_gate(
        incumbent,
        candidate,
        (replace(units[0], candidate=replace(new_target, invalid_repeated_write_calls=None)),
         *units[1:]),
        target_failure=failure,
        proposal=proposal,
        audit=audit,
        initial_service_strategy_id=old_service_id,
        token_counter=lambda text: len(text.split()),
    )
    assert not missing_repeat_audit.accepted
    assert any("missing invalid repeated-write counts" in reason for reason in missing_repeat_audit.reasons)

    incumbent_regressed_clean = replace(old_clean, episode_id="old-clean-regressed", task_success=False)
    candidate_still_regressed_clean = replace(
        new_clean, episode_id="new-clean-regressed", task_success=False,
    )
    anchor_regression = evaluate_repair_gate(
        incumbent,
        candidate,
        units[:2] + (
            GateUnit("clean", "clean", incumbent_regressed_clean,
                     candidate_still_regressed_clean, initial_s0=initial_s0_clean),
        ) + units[3:],
        target_failure=failure,
        proposal=proposal,
        audit=audit,
        initial_service_strategy_id=old_service_id,
        token_counter=lambda text: len(text.split()),
    )
    assert not anchor_regression.accepted
    assert any("initial S0" in reason for reason in anchor_regression.reasons)

    missing_anchor = evaluate_repair_gate(
        incumbent,
        candidate,
        units[:2] + (GateUnit("clean", "clean", old_clean, new_clean),) + units[3:],
        target_failure=failure,
        proposal=proposal,
        audit=audit,
        initial_service_strategy_id=old_service_id,
        token_counter=lambda text: len(text.split()),
    )
    assert not missing_anchor.accepted
    assert any("missing its initial S0" in reason for reason in missing_anchor.reasons)

    wrong_s0 = replace(initial_s0_clean, service_strategy_id="wrong-initial-checkpoint")
    misbound_anchor = evaluate_repair_gate(
        incumbent,
        candidate,
        units[:2] + (
            GateUnit("clean", "clean", old_clean, new_clean, initial_s0=wrong_s0),
        ) + units[3:],
        target_failure=failure,
        proposal=proposal,
        audit=audit,
        initial_service_strategy_id=old_service_id,
        token_counter=lambda text: len(text.split()),
    )
    assert not misbound_anchor.accepted
    assert any("frozen Service checkpoint ID" in reason for reason in misbound_anchor.reasons)

    restored_anchor = evaluate_repair_gate(
        incumbent,
        candidate,
        units[:2] + (
            GateUnit("clean", "clean", incumbent_regressed_clean, new_clean,
                     initial_s0=initial_s0_clean),
        ) + units[3:],
        target_failure=failure,
        proposal=proposal,
        audit=audit,
        initial_service_strategy_id=old_service_id,
        token_counter=lambda text: len(text.split()),
    )
    assert restored_anchor.accepted


def test_checkpoint_atomic_roundtrip_and_manifest_binding(tmp_path):
    manifest = {"experiment": "fixture", "seed": 4}
    digest = manifest_fingerprint(manifest)
    path = tmp_path / "state.json"
    save_checkpoint(path, EvolutionCheckpoint(digest, 0, {"value": 7}, ("generation:0",)))
    assert load_checkpoint(path, expected_manifest_hash=digest).state == {"value": 7}
    with pytest.raises(ValueError, match="manifest"):
        load_checkpoint(path, expected_manifest_hash=manifest_fingerprint({"experiment": "changed"}))


def test_request_budget_restores_cumulative_attempt_accounting():
    budget = RequestBudget(10)
    budget.restore_usage(BudgetSnapshot(
        cap=10, attempts=3, successes=2, failures=1, denied=1,
        prompt_tokens=20, completion_tokens=10,
        usage_responses=2, usage_unavailable=1, cache_hits=1,
    ))
    assert budget.snapshot().remaining == 7
    assert budget.snapshot().prompt_tokens == 20
    assert budget.snapshot().completion_tokens == 10
    assert budget.snapshot().cache_hits == 1
    with pytest.raises(ValueError, match="before it is used"):
        budget.restore_usage(BudgetSnapshot(cap=10, attempts=1, successes=1, failures=0, denied=0))


def test_global_budget_absorbs_child_phase_usage_and_enforces_total_cap():
    total = RequestBudget(8)
    total.absorb_usage(BudgetSnapshot(
        cap=70, attempts=3, successes=2, failures=1, denied=0,
        prompt_tokens=12, completion_tokens=7,
        usage_responses=2, usage_unavailable=1, cache_hits=0,
    ))
    assert total.snapshot().attempts == 3
    assert total.snapshot().remaining == 5
    assert total.snapshot().prompt_tokens == 12
    with pytest.raises(ValueError, match="exceeds"):
        total.absorb_usage(BudgetSnapshot(
            cap=70, attempts=6, successes=6, failures=0, denied=0,
        ))


def test_phase3_manifest_freezes_tasks_generations_and_hard_caps():
    config_path = __import__("pathlib").Path(__file__).parents[1] / "configs" / "phase3-mechanism.yaml"
    manifest = MechanismManifest.from_mapping(load_config(config_path))
    assert manifest.evolution_task_id == "73"
    assert manifest.validation_task_id == "93"
    assert manifest.max_episodes == 23 and manifest.request_budget_cap == 1800
    assert not manifest.real_provider_enabled
    payload = manifest.to_payload()
    assert payload["failure_taxonomy"] == [
        {"workflow_stage": stage, "policy_rule_id": policy_rule_id, "mistake_type": mistake}
        for stage, policy_rule_id, mistake in MVP_FAILURE_TAXONOMY
    ]
    assert payload["failure_taxonomy_sha256"] == sha256_json(MVP_FAILURE_TAXONOMY)
    assert manifest.sha256 == manifest_fingerprint(payload)
    invalid = load_config(config_path)
    invalid["experiment"]["request_budget_cap"] = 1801
    with pytest.raises(ValueError, match="1800"):
        MechanismManifest.from_mapping(invalid)


def test_manifest_bound_controller_uses_a_fresh_e_seed_for_confirmation(tmp_path):
    config_path = __import__("pathlib").Path(__file__).parents[1] / "configs" / "phase3-mechanism.yaml"
    manifest = MechanismManifest.from_mapping(load_config(config_path))
    customer, service = CustomerStrategy(), ServiceStrategy()
    winner = propose_customer_candidates(customer, 2, seed=manifest.seed)[0]
    calls = []

    def runner(*, task_id, seed, customer, service, panel_name):
        calls.append((task_id, seed, panel_name))
        is_winner = customer_strategy_id(customer) == winner.strategy_id
        return EpisodeRecord(
            episode_id=f"{panel_name}:{task_id}:{seed}:{customer_strategy_id(customer)}",
            task_id=task_id,
            seed=seed,
            customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service),
            status=EpisodeStatus.COMPLETE,
            task_success=not is_winner,
            customer_valid=True,
            customer_strategy_adherent=True,
            policy_violation=is_winner,
            invalid_repeated_write_calls=0,
            policy_rule_id="retail.policy:explicit_confirmation",
            mistake_type="missing_explicit_confirmation",
            workflow_stage="pre_write",
            evidence=(EvidenceRef(2, "tool", "write occurred before confirmation"),) if is_winner else (),
        )

    controller = TwoGenerationSmoke(
        manifest=manifest,
        checkpoint_path=str(tmp_path / "fresh-confirmation.json"),
        runner=runner,
        task_ids=(manifest.evolution_task_id, manifest.validation_task_id),
        seed=manifest.seed,
    )
    service_customers = []

    def no_repair(generation, evolved_customer, current_service, failures, episode_runner, budget):
        assert failures
        assert episode_runner is not None
        assert budget is None
        service_customers.append(customer_strategy_id(evolved_customer))
        return current_service, None, "no service proposal in this fixture"

    commits = controller.run(
        customer,
        service,
        failure_verifier=lambda item: "audit:independent-fixture" if item.policy_violation else None,
        service_transition=no_repair,
    )
    assert commits[0].customer_evolved
    assert commits[0].customer_id == winner.strategy_id
    assert (manifest.evolution_task_id, manifest.seed + 10_000, "confirmation") in calls
    assert not any(panel == "confirmation" and task == manifest.validation_task_id
                   for task, _seed, panel in calls)
    assert service_customers == [winner.strategy_id, winner.strategy_id]


def test_manifest_bound_controller_rejects_unfrozen_initial_strategy(tmp_path):
    config_path = __import__("pathlib").Path(__file__).parents[1] / "configs" / "phase3-mechanism.yaml"
    manifest = MechanismManifest.from_mapping(load_config(config_path))
    controller = TwoGenerationSmoke(
        manifest=manifest,
        checkpoint_path=str(tmp_path / "mismatched-strategy.json"),
        runner=lambda **kwargs: None,
        task_ids=(manifest.evolution_task_id, manifest.validation_task_id),
        seed=manifest.seed,
    )
    with pytest.raises(ValueError, match="initial Customer strategy"):
        controller.run(CustomerStrategy(disclosure="related_on_request"), ServiceStrategy())


def test_two_generation_smoke_commits_are_idempotent_and_detect_manifest_drift(tmp_path):
    manifest = {"experiment": "fixture"}
    smoke = TwoGenerationSmoke(manifest=manifest, checkpoint_path=str(tmp_path / "state.json"),
                               runner=lambda **kwargs: None, task_ids=("E", "V"), seed=1)
    c, s = CustomerStrategy(), ServiceStrategy()
    first = smoke.commit_generation(0, c, s)
    assert smoke.commit_generation(0, c, s) == first
    smoke.commit_generation(1, c, s, note="no replacement is valid")
    loaded = load_checkpoint(tmp_path / "state.json", expected_manifest_hash=manifest_fingerprint(manifest))
    assert loaded.generation == 1
    assert len(loaded.state["commits"]) == 2
    changed = TwoGenerationSmoke(manifest={"experiment": "changed"},
        checkpoint_path=str(tmp_path / "state.json"), runner=lambda **kwargs: None,
        task_ids=("E", "V"), seed=1)
    with pytest.raises(ValueError, match="manifest"):
        changed.commit_generation(1, c, s)
    fresh = TwoGenerationSmoke(manifest={"experiment": "fresh"},
        checkpoint_path=str(tmp_path / "fresh.json"), runner=lambda **kwargs: None,
        task_ids=("E", "V"), seed=1)
    with pytest.raises(ValueError, match="start at generation 0"):
        fresh.commit_generation(1, c, s)


def test_two_generation_controller_runs_customer_first_and_commits_no_change(tmp_path):
    customer, service = CustomerStrategy(), ServiceStrategy()

    def runner(*, task_id, seed, customer, service, panel_name):
        return EpisodeRecord(
            episode_id=f"{panel_name}:{task_id}:{seed}:{customer_strategy_id(customer)}",
            task_id=task_id, seed=seed, customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service), status=EpisodeStatus.COMPLETE,
            task_success=True, customer_valid=True, customer_strategy_adherent=True,
            invalid_repeated_write_calls=0,
        )

    controller = TwoGenerationSmoke(manifest={"fixture": True}, checkpoint_path=str(tmp_path / "run.json"),
                                   runner=runner, task_ids=("E", "V"), seed=3)
    commits = controller.run(customer, service)
    assert len(commits) == 2
    assert all(commit.completed and not commit.customer_evolved and not commit.service_evolved for commit in commits)
    state = load_checkpoint(tmp_path / "run.json", expected_manifest_hash=manifest_fingerprint({"fixture": True}))
    assert len(state.state["commits"]) == 2
    calls = 0

    def counted_runner(**kwargs):
        nonlocal calls
        calls += 1
        return runner(**kwargs)

    resumed = TwoGenerationSmoke(manifest={"fixture": True}, checkpoint_path=str(tmp_path / "run.json"),
                                 runner=counted_runner, task_ids=("E", "V"), seed=3)
    assert resumed.run(customer, service) == commits
    assert calls == 0


def test_mid_generation_resume_reuses_completed_episode_records(tmp_path):
    customer, service = CustomerStrategy(), ServiceStrategy()
    dispatches = 0
    failed_once = False

    def runner(*, task_id, seed, customer, service, panel_name):
        nonlocal dispatches, failed_once
        dispatches += 1
        if dispatches == 2 and not failed_once:
            failed_once = True
            raise RuntimeError("simulated process interruption")
        return EpisodeRecord(
            episode_id=f"{panel_name}:{task_id}:{seed}:{customer_strategy_id(customer)}",
            task_id=task_id, seed=seed, customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service), status=EpisodeStatus.COMPLETE,
            task_success=True, customer_valid=True, customer_strategy_adherent=True,
            invalid_repeated_write_calls=0,
        )

    path = str(tmp_path / "resume.json")
    manifest = {"fixture": "resume"}
    controller = TwoGenerationSmoke(manifest=manifest, checkpoint_path=path, runner=runner,
                                    task_ids=("E", "V"), seed=9)
    with pytest.raises(RuntimeError, match="interruption"):
        controller.run(customer, service)
    saved = load_checkpoint(path, expected_manifest_hash=manifest_fingerprint(manifest))
    assert len(saved.state["progress"]["episodes"]) == 1
    resumed = TwoGenerationSmoke(manifest=manifest, checkpoint_path=path, runner=runner,
                                  task_ids=("E", "V"), seed=9)
    resumed.run(customer, service)
    assert dispatches == 7  # the one completed pre-interruption episode is served from checkpoint
    final = load_checkpoint(path, expected_manifest_hash=manifest_fingerprint(manifest))
    assert final.state["episode_attempts"] == 7


def test_pilot_controller_runs_three_generations_over_frozen_evolution_panel(tmp_path):
    customer, service = CustomerStrategy(), ServiceStrategy()
    calls = []

    def runner(*, task_id, seed, customer, service, panel_name):
        calls.append((task_id, seed, panel_name))
        return EpisodeRecord(
            episode_id=f"{panel_name}:{task_id}:{seed}:{customer_strategy_id(customer)}",
            task_id=task_id, seed=seed, customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service), status=EpisodeStatus.COMPLETE,
            task_success=True, customer_valid=True, customer_strategy_adherent=True,
            invalid_repeated_write_calls=0,
        )

    manifest = {"fixture": True, "generations": 3, "max_episodes": 30}
    controller = TwoGenerationSmoke(
        manifest=manifest, checkpoint_path=str(tmp_path / "pilot-run.json"),
        runner=runner, task_ids=("E1", "V1"), evolution_task_ids=("E1", "E2", "E3"),
        seed=19,
    )
    commits = controller.run(customer, service)

    assert [item.generation for item in commits] == [0, 1, 2]
    assert len(calls) == 27
    assert {item[0] for item in calls} == {"E1", "E2", "E3"}
    assert {item[1] for item in calls} == {19, 20, 21}
    assert all(item[0] != "V1" for item in calls)
    resumed = TwoGenerationSmoke(
        manifest=manifest, checkpoint_path=str(tmp_path / "pilot-run.json"),
        runner=runner, task_ids=("E1", "V1"), evolution_task_ids=("E1", "E2", "E3"),
        seed=19,
    )
    assert resumed.run(customer, service) == commits
    assert len(calls) == 27
    final = load_checkpoint(
        tmp_path / "pilot-run.json",
        expected_manifest_hash=manifest_fingerprint(manifest),
    )
    assert final.state["episode_attempts"] == 27
    assert len(final.state["commits"]) == 3


def test_controller_uses_manifest_concurrency_across_frozen_candidate_batch(tmp_path):
    customer, service = CustomerStrategy(), ServiceStrategy()
    lock = Lock()
    active = 0
    max_active = 0
    calls = []

    def runner(*, task_id, seed, customer, service, panel_name):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            calls.append((task_id, seed, customer_strategy_id(customer), panel_name))
        sleep(0.004)
        with lock:
            active -= 1
        return EpisodeRecord(
            episode_id=f"{panel_name}:{task_id}:{seed}:{customer_strategy_id(customer)}",
            task_id=task_id, seed=seed, customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service), status=EpisodeStatus.COMPLETE,
            task_success=True, customer_valid=True, customer_strategy_adherent=True,
            invalid_repeated_write_calls=0,
        )

    manifest = {
        "fixture": "concurrent-controller", "generations": 2,
        "max_episodes": 30, "max_concurrency": 4,
    }
    controller = TwoGenerationSmoke(
        manifest=manifest, checkpoint_path=str(tmp_path / "concurrent-run.json"),
        runner=runner, task_ids=("E1", "V1"), evolution_task_ids=("E1", "E2", "E3", "E4"),
        seed=11,
    )
    commits = controller.run(customer, service)
    assert [item.generation for item in commits] == [0, 1]
    assert max_active == 4
    assert len(calls) == 24
    assert {item[0] for item in calls} == {"E1", "E2", "E3", "E4"}


def test_concurrent_panel_error_checkpoints_drained_results_for_resume(tmp_path):
    customer, service = CustomerStrategy(), ServiceStrategy()
    started = Barrier(4)
    lock = Lock()
    failed_once = False
    attempts_by_key = {}

    def runner(*, task_id, seed, customer, service, panel_name):
        nonlocal failed_once
        key = (task_id, seed, customer_strategy_id(customer), panel_name)
        with lock:
            attempts_by_key[key] = attempts_by_key.get(key, 0) + 1
            should_sync = (
                not failed_once
                and panel_name == "discovery"
                and key[2] == customer_strategy_id(CustomerStrategy())
            )
        if should_sync:
            started.wait(timeout=3)
        if task_id == "E1" and key[2] == customer_strategy_id(CustomerStrategy()) and not failed_once:
            with lock:
                if not failed_once:
                    failed_once = True
                    raise RuntimeError("one worker interrupted")
        return EpisodeRecord(
            episode_id=f"{panel_name}:{task_id}:{seed}:{key[2]}",
            task_id=task_id, seed=seed, customer_strategy_id=key[2],
            service_strategy_id=service_strategy_id(service), status=EpisodeStatus.COMPLETE,
            task_success=True, customer_valid=True, customer_strategy_adherent=True,
            invalid_repeated_write_calls=0,
        )

    manifest = {
        "fixture": "concurrent-resume", "generations": 2,
        "max_episodes": 40, "max_concurrency": 4,
    }
    path = str(tmp_path / "concurrent-resume.json")
    controller = TwoGenerationSmoke(
        manifest=manifest, checkpoint_path=path, runner=runner,
        task_ids=("E1", "V1"), evolution_task_ids=("E1", "E2", "E3", "E4"), seed=12,
    )
    with pytest.raises(RuntimeError, match="one worker interrupted"):
        controller.run(customer, service)
    saved = load_checkpoint(path, expected_manifest_hash=manifest_fingerprint(manifest))
    completed_keys = saved.state["progress"]["episodes"]
    assert len(completed_keys) == 3
    assert saved.state["episode_attempts"] == 4

    resumed = TwoGenerationSmoke(
        manifest=manifest, checkpoint_path=path, runner=runner,
        task_ids=("E1", "V1"), evolution_task_ids=("E1", "E2", "E3", "E4"), seed=12,
    )
    resumed.run(customer, service)
    incumbent_id = customer_strategy_id(customer)
    assert [attempts_by_key[(task_id, 12, incumbent_id, "discovery")]
            for task_id in ("E1", "E2", "E3", "E4")] == [2, 1, 1, 1]


def test_serial_controller_checkpoint_releases_unused_provider_reservations(tmp_path):
    customer, service = CustomerStrategy(), ServiceStrategy()
    budget = RequestBudget(cap=20)
    provider = SimpleNamespace(
        completion=lambda **_kwargs: SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        ),
        DEFAULT_MAX_RETRIES=0,
    )

    def runner(*, task_id, seed, customer, service, panel_name):
        with budget.instrument_tau_llm_utils(provider):
            provider.completion(model="mock")
        return EpisodeRecord(
            episode_id=f"{panel_name}:{task_id}:{seed}:{customer_strategy_id(customer)}",
            task_id=task_id, seed=seed, customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service), status=EpisodeStatus.COMPLETE,
            task_success=True, customer_valid=True, customer_strategy_adherent=True,
            invalid_repeated_write_calls=0,
        )

    manifest = {"fixture": "serial-budget-checkpoint", "max_episodes": 20}
    controller = TwoGenerationSmoke(
        manifest=manifest, checkpoint_path=str(tmp_path / "serial-budget.json"),
        runner=runner, task_ids=("E1", "V1"), evolution_task_ids=("E1", "E2", "E3"),
        seed=13, request_budget=budget,
    )
    controller.run(customer, service)
    saved = load_checkpoint(
        tmp_path / "serial-budget.json", expected_manifest_hash=manifest_fingerprint(manifest),
    )
    assert saved.state["request_budget"]["attempts"] == 18
    assert saved.state["request_budget"]["reserved"] == 0
    assert saved.state["request_budget"]["in_flight"] == 0


def test_prepared_generation_resume_reuses_persisted_selection_and_service_decision(tmp_path):
    customer, service = CustomerStrategy(), ServiceStrategy()
    dispatches = 0
    service_transitions = 0

    def runner(*, task_id, seed, customer, service, panel_name):
        nonlocal dispatches
        dispatches += 1
        customer_id = customer_strategy_id(customer)
        service_id = service_strategy_id(service)
        return episode(
            name=f"{panel_name}:{task_id}:{seed}:{customer_id}", task=task_id, seed=seed,
            customer=customer_id, service=service_id,
        )

    def transition(_generation, _customer, current_service, failures, _runner, _budget):
        nonlocal service_transitions
        service_transitions += 1
        assert failures
        failure = failures[0]
        rule = ServiceRule(
            "r1", failure.policy_ref, "before a database write",
            "summarize all action details and get explicit yes confirmation before writing",
            failure.evidence_refs,
        )
        proposal = RepairProposal(
            failure.failure_id, rule,
            "The same eligible write succeeds only after explicit confirmation is collected.",
        )
        audit = RepairAudit(
            True, "audit:durability-fixture", frozenset({failure.policy_ref}),
            rationale="The proposal restates the fixed policy and adds no permission.",
        )
        evaluated_candidate = build_repair_candidate(
            current_service, proposal, failure, audit,
            token_counter=lambda text: len(text.split()),
        )
        rejected_gate = GateReport(
            accepted=False,
            reasons=("fixture clean regression",),
            candidate_strategy=None,
            unit_results=(("clean", False, "candidate regressed clean validation"),),
            target_failure_id=failure.failure_id,
            proposal=proposal,
            audit=audit,
            evaluated_candidate=evaluated_candidate,
        )
        return current_service, rejected_gate, "repair rejected by clean gate"

    class InterruptBeforeCommit(TwoGenerationSmoke):
        def commit_generation(self, generation, *args, **kwargs):
            if generation == 0:
                raise RuntimeError("crash after decision journal, before commit")
            return super().commit_generation(generation, *args, **kwargs)

    checkpoint = tmp_path / "prepared.json"
    archive_path = tmp_path / "archive.sqlite"
    manifest = {"fixture": "prepared-generation"}
    archive = FailureArchive(archive_path)
    first = InterruptBeforeCommit(
        manifest=manifest, checkpoint_path=str(checkpoint), runner=runner,
        task_ids=("E", "V"), seed=9, failure_archive=archive,
    )
    with pytest.raises(RuntimeError, match="decision journal"):
        first.run(
            customer, service,
            failure_verifier=lambda _item: "audit:durability-fixture",
            service_transition=transition,
        )

    prepared = load_checkpoint(checkpoint, expected_manifest_hash=manifest_fingerprint(manifest))
    prepared_generation = prepared.state["progress"]["prepared_generation"]
    assert prepared_generation["commit"]["generation"] == 0
    decision = prepared_generation["commit"]["decision_record"]
    assert decision["customer"]["selection"]["evolved"] is False
    assert len(decision["customer"]["evaluations"]) == 4  # discovery + one baseline replay
    assert decision["service"]["transition_ran"] is True
    assert decision["service"]["gate"]["accepted"] is False
    assert decision["service"]["gate"]["accepted_candidate"] is None
    assert decision["service"]["gate"]["evaluated_candidate"]["rules"][0]["rule_id"] == "r1"
    assert decision["service"]["gate"]["proposal"]["verification_hypothesis"].startswith(
        "The same eligible write"
    )
    assert decision["service"]["gate"]["audit"]["verifier_ref"] == "audit:durability-fixture"
    assert len(decision["verified_failures"]) >= 1
    assert decision["active_replay_coverage"]["active_signatures"] == 1
    assert decision["active_replay_coverage"]["uncovered_signatures"] == 1
    assert service_transitions == 1
    assert dispatches == 4

    resumed = TwoGenerationSmoke(
        manifest=manifest, checkpoint_path=str(checkpoint), runner=runner,
        task_ids=("E", "V"), seed=9, failure_archive=FailureArchive(archive_path),
    )
    commits = resumed.run(
        customer, service,
        failure_verifier=lambda _item: "audit:durability-fixture",
        service_transition=transition,
    )
    assert tuple(item.generation for item in commits) == (0, 1)
    assert service_transitions == 2  # only generation 1 is newly evaluated after resume
    assert dispatches == 7  # generation 0's four cached episodes are not rerun
    final = load_checkpoint(checkpoint, expected_manifest_hash=manifest_fingerprint(manifest))
    assert final.state["progress"] is None
    assert final.state["commits"][0]["decision_record"] == decision


def test_controller_applies_episode_and_shared_request_caps(tmp_path):
    invoked = 0

    def runner(**kwargs):
        nonlocal invoked
        invoked += 1
        raise AssertionError("runner must not be dispatched after cap")

    c, s = CustomerStrategy(), ServiceStrategy()
    controller = TwoGenerationSmoke(manifest={"max_episodes": 0},
        checkpoint_path=str(tmp_path / "capped.json"), runner=runner,
        task_ids=("E", "V"), seed=1)
    with pytest.raises(RuntimeError, match="episode cap"):
        controller.run(c, s)
    assert invoked == 0
    with pytest.raises(ValueError, match="1,800"):
        TwoGenerationSmoke(manifest={}, checkpoint_path=str(tmp_path / "too-large.json"),
                          runner=runner, task_ids=("E", "V"), seed=1,
                          request_budget=RequestBudget(1801))


def test_phase0_context_counts_toward_manifest_episode_cap(tmp_path):
    invoked = False

    def runner(**kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("Phase 0 already consumed the one-episode cap")

    controller = TwoGenerationSmoke(
        manifest={"max_episodes": 1}, checkpoint_path=str(tmp_path / "phase3.json"),
        runner=runner, task_ids=("E", "V"), seed=1,
        manifest_context={"phase0_result_sha256": "fixture"},
    )
    assert controller.episode_attempts == 1
    with pytest.raises(RuntimeError, match="episode cap"):
        controller.run(CustomerStrategy(), ServiceStrategy())
    assert not invoked


def test_generation_lifecycle_commits_only_independently_verified_discoveries(tmp_path):
    customer, service = CustomerStrategy(), ServiceStrategy()
    archive = FailureArchive(tmp_path / "history.sqlite")

    def runner(*, task_id, seed, customer, service, panel_name):
        valid_failure = episode(name=f"{panel_name}:{task_id}:{seed}", task=task_id, seed=seed,
                                customer=customer_strategy_id(customer), service=service_strategy_id(service))
        return replace(valid_failure, task_success=False)

    controller = TwoGenerationSmoke(manifest={"audit": "fixture"},
        checkpoint_path=str(tmp_path / "audited-run.json"), runner=runner,
        task_ids=("E", "V"), seed=5, failure_archive=archive)
    # Every failure trace receives its own independent audit artifact.

    def no_repair(_generation, _customer, current_service, failures, _episode_runner, budget):
        assert failures
        assert budget is None
        return current_service, None, "no repair proposal in fixture"

    controller.run(
        customer, service,
        failure_verifier=lambda item: f"audit:{item.episode_id}",
        service_transition=no_repair,
    )
    assert archive.recent()
    assert archive.customer_strategies()
    assert archive.service_strategies()
