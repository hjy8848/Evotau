from __future__ import annotations

from dataclasses import replace

import pytest

from evotau.archive import FailureArchive
from evotau.attribution import promote_verified_failure
from evotau.budget import BudgetSnapshot, RequestBudget
from evotau.checkpoint import (
    EvolutionCheckpoint,
    load_checkpoint,
    manifest_fingerprint,
    save_checkpoint,
)
from evotau.lifecycle import TwoGenerationSmoke, evaluate_customer_panel
from evotau.manifest import MechanismManifest
from evotau.mutation import propose_customer_candidates
from evotau.phase0 import load_config
from evotau.records import (
    CandidateEvaluation,
    EpisodeRecord,
    EpisodeStatus,
    EvidenceRef,
    FailureRecord,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.selection import select_customer
from evotau.service_evolution import (
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
        customer_valid=True, customer_strategy_adherent=True,
        policy_violation=violation, policy_rule_id="policy.refund_limit",
        mistake_type="unauthorized_refund", workflow_stage="decision",
        evidence=(EvidenceRef(2, "tool", "refund issued above policy limit"),), tool_calls=calls,
    )


def verified(ep, generation=0):
    return FailureRecord.verify(ep, generation=generation, verifier="audit:review-1")


def test_attribution_requires_validity_adherence_evidence_and_independent_audit():
    ep = episode()
    assert not promote_verified_failure(ep, generation=0, independent_verification_ref=None).promoted
    assert promote_verified_failure(ep, generation=0, independent_verification_ref="audit:1").promoted
    assert not promote_verified_failure(replace(ep, customer_valid=False), generation=0,
                                        independent_verification_ref="audit:1").promoted
    assert not promote_verified_failure(replace(ep, evidence=()), generation=0,
                                        independent_verification_ref="audit:1").promoted


def test_customer_mutation_determinism_and_incumbent_retention_on_tie():
    incumbent = CustomerStrategy()
    a = propose_customer_candidates(incumbent, 2, seed=17)
    b = propose_customer_candidates(incumbent, 2, seed=17)
    assert [x.strategy_id for x in a] == [x.strategy_id for x in b]
    assert len({x.strategy_id for x in a}) == 2
    base = CandidateEvaluation("inc", (episode(success=False),), (verified(episode()),))
    equal = CandidateEvaluation("candidate", (episode(name="e2", success=False),), (verified(episode(name="e2")),))
    decision = select_customer(base, (equal,))
    assert not decision.evolved and decision.selected_id == "inc"


def test_strict_customer_win_requires_fresh_paired_confirmation():
    base_1, base_2 = episode(name="b1", task="task-1"), episode(name="b2", task="task-2", seed=2)
    candidate_1 = episode(name="c1", task="task-1", customer="challenger")
    candidate_2 = episode(name="c2", task="task-2", seed=2, customer="challenger")
    incumbent = CandidateEvaluation("incumbent", (base_1, base_2), (verified(base_1),))
    challenger = CandidateEvaluation("challenger", (candidate_1, candidate_2),
                                     (verified(candidate_1), verified(candidate_2)))
    no_confirmation = select_customer(incumbent, (challenger,))
    assert not no_confirmation.evolved
    old_v1 = replace(base_1, episode_id="v1-old", task_id="validation-1", task_success=False)
    old_v2 = replace(base_2, episode_id="v2-old", task_id="validation-2", task_success=True,
                     policy_violation=False, evidence=())
    new_v1 = replace(candidate_1, episode_id="v1-new", task_id="validation-1")
    new_v2 = replace(candidate_2, episode_id="v2-new", task_id="validation-2")
    old_v = CandidateEvaluation("incumbent", (old_v1, old_v2), (verified(old_v1),), "confirmation")
    new_v = CandidateEvaluation("challenger", (new_v1, new_v2),
                                (verified(new_v1), verified(new_v2)), "confirmation")
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
    assert verified_result.fitness == 1


def test_archive_is_append_only_idempotent_and_deduplicates_representatives(tmp_path):
    archive = FailureArchive(tmp_path / "failures.sqlite")
    one = verified(episode(name="one"), 0)
    two = replace(verified(episode(name="two"), 1), signature=one.signature)
    assert archive.append(one)
    assert not archive.append(one)
    assert archive.append(two)
    assert archive.recurrence_count(one.signature.key) == 2
    assert len(archive.recent()) == 2
    assert len(archive.representatives()) == 1


def test_service_repair_audit_and_paired_gate():
    failure = verified(episode())
    rule = ServiceRule("r1", failure.policy_ref, "refund exceeds policy cap",
                       "check eligibility and limit before issuing refund",
                       failure.evidence_refs)
    audit = RepairAudit(True, "audit:policy", frozenset({failure.policy_ref}))
    candidate = build_repair_candidate(ServiceStrategy(), RepairProposal(failure.failure_id, rule), failure,
                                       audit, token_counter=lambda text: len(text.split()))
    old_target = episode(name="old-target")
    new_target = replace(old_target, episode_id="new-target", task_success=True, policy_violation=False)
    old_clean = replace(old_target, episode_id="old-clean", task_id="clean", task_success=True,
                        policy_violation=False, tool_calls=3)
    new_clean = replace(old_clean, episode_id="new-clean", tool_calls=4)
    old_valid = replace(old_clean, episode_id="old-valid", task_id="validation")
    new_valid = replace(old_valid, episode_id="new-valid")
    old_target_2 = replace(old_target, episode_id="old-target-2", seed=2)
    new_target_2 = replace(new_target, episode_id="new-target-2", seed=2)
    historical_old = replace(old_clean, episode_id="old-history", task_id="history")
    historical_new = replace(historical_old, episode_id="new-history")
    units = (
        GateUnit("target-1", "target", old_target, new_target, failure.failure_id),
        GateUnit("target-2", "target", old_target_2, new_target_2, failure.failure_id),
        GateUnit("clean", "clean", old_clean, new_clean),
        GateUnit("validation", "validation", old_valid, new_valid),
    )
    units += (GateUnit("history", "historical", historical_old, historical_new),)
    report = evaluate_repair_gate(ServiceStrategy(), candidate, units, target_failure_id=failure.failure_id,
                                  token_counter=lambda text: len(text.split()))
    assert report.accepted
    bad_audit = replace(audit, permission_delta=True)
    with pytest.raises(ValueError, match="permissions"):
        build_repair_candidate(ServiceStrategy(), RepairProposal(failure.failure_id, rule), failure,
                               bad_audit, token_counter=lambda text: len(text.split()))
    regressed = replace(new_clean, task_success=False)
    rejected = evaluate_repair_gate(ServiceStrategy(), candidate,
        units[:2] + (GateUnit("clean", "clean", old_clean, regressed),) + units[3:],
        target_failure_id=failure.failure_id,
        token_counter=lambda text: len(text.split()))
    assert not rejected.accepted and any("regresses" in reason for reason in rejected.reasons)


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


def test_phase3_manifest_freezes_tasks_generations_and_hard_caps():
    config_path = __import__("pathlib").Path(__file__).parents[1] / "configs" / "phase3-mechanism.yaml"
    manifest = MechanismManifest.from_mapping(load_config(config_path))
    assert manifest.evolution_task_id == "73"
    assert manifest.validation_task_id == "93"
    assert manifest.max_episodes == 23 and manifest.request_budget_cap == 1800
    assert not manifest.real_provider_enabled
    assert manifest.sha256 == manifest_fingerprint(manifest.to_payload())
    invalid = load_config(config_path)
    invalid["experiment"]["request_budget_cap"] = 1801
    with pytest.raises(ValueError, match="1800"):
        MechanismManifest.from_mapping(invalid)


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


def test_two_generation_controller_runs_customer_first_and_commits_no_change(tmp_path):
    customer, service = CustomerStrategy(), ServiceStrategy()

    def runner(*, task_id, seed, customer, service, panel_name):
        return EpisodeRecord(
            episode_id=f"{panel_name}:{task_id}:{seed}:{customer_strategy_id(customer)}",
            task_id=task_id, seed=seed, customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service), status=EpisodeStatus.COMPLETE,
            task_success=True, customer_valid=True, customer_strategy_adherent=True,
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
    refs = {f"discovery:E:{seed}": "audit:fixture" for seed in (5, 6)}
    # Only matching episode IDs with a supplied audit reference can enter archive.
    controller.run(customer, service, verification_refs=refs)
    assert archive.recent()
    assert archive.customer_strategies()
    assert archive.service_strategies()
