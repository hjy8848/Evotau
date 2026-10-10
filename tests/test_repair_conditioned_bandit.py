"""Scripted RC tests: no model/API/native scoring substitutes in production."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from test_customer_skill_protocol import SkillProviders, policy, skill
from test_skill_evolution_v2 import FakeRunner, mutation, run

from evotau.customer_skills import CustomerSkill
from evotau.evolution_artifacts import EvolutionJournal
from evotau.records import customer_strategy_id
from evotau.repair_conditioned_bandit import (
    ARMS,
    DEFAULT_POLICY,
    PROTOCOL,
    choose_arm,
    new_state,
    record_trial,
    restore,
    serialize,
    validate_policy,
)
from evotau.repair_feedback import (
    discovery_feedback,
    paired_observations,
    review_context,
    service_pair,
    validate_review,
)
from evotau.service_skills import ServiceSkillMemoryV2, apply_v2_mutation
from evotau.skill_evolution_config import freeze_v2_policy
from evotau.tau_provenance import sha256_json


def pair_fixture():
    before = ServiceSkillMemoryV2()
    after = apply_v2_mutation(before, mutation(), next_skill_id_number=1)
    return service_pair(before, after, generation=0, promoted=True)


def samples():
    from test_skill_evolution_v2 import record

    pair = pair_fixture()
    c = CustomerSkill.from_mapping(skill()).compile()
    left, right = [], []
    for t, a, b in (
        ("1", False, True),
        ("2", False, False),
        ("3", True, False),
        ("4", True, True),
    ):
        for seed in (1, 2):
            for out, ok, endpoint in ((left, a, "before"), (right, b, "after")):
                r = record(
                    t, ok, seed, trajectory_ref=f"scripted/{endpoint}/{t}/{seed}"
                )
                out.append(
                    replace(
                        r,
                        customer_strategy_id=customer_strategy_id(c),
                        service_strategy_id=pair[endpoint + "_id"],
                        episode_id=f"{endpoint}-{t}-{seed}",
                    )
                )
    validity = {
        name: {
            "status": "valid",
            "cells": [
                {
                    "task_id": t,
                    "seed": s,
                    "status": "valid",
                    "reason": "scripted preserved facts",
                    "evidence_message_indices": [0],
                }
                for t in ("1", "2", "3", "4")
                for s in (1, 2)
            ],
        }
        for name in ("before", "after")
    }
    return pair, c, left, right, validity


def observations():
    pair, c, left, right, validity = samples()
    obs = paired_observations(
        left,
        right,
        pair=pair,
        customer_id=customer_strategy_id(c),
        task_ids=["1", "2", "3", "4"],
        seeds=[1, 2],
        validity=validity,
    )
    rows = {}
    for name, records in (("before", left), ("after", right)):
        rows[name] = [
            {
                "task": {"task_id": r.task_id, "user_scenario": "NEVER_COPY_HIDDEN"},
                "seed": r.seed,
                "trajectory_ref": r.trajectory_ref,
                "trajectory": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "scripted action evidence",
                            "evidence_ref": {
                                "projected_message_index": 1,
                                "message_sha256": sha256_json(
                                    [r.task_id, r.seed, name]
                                ),
                            },
                        }
                    ]
                },
            }
            for r in records
        ]
    ctx = review_context(obs, pair, rows, prior_discoveries=[])
    return obs, ctx


def supported_review(ctx, task="2", kind="residual", seeds=(1, 2)):
    return deepcopy(
        {
            "discoveries": [
                {
                    "mechanism": "Scripted missing conditional confirmation",
                    "task_id": task,
                    "discovery_type": kind,
                    "seeds": list(seeds),
                    "repair_related": True,
                    "novel": True,
                    "reason": "Scripted mock evidence, not a research finding.",
                    "evidence_refs": [
                        m["evidence_ref"]
                        for r in ctx["evidence"]
                        if r["task_id"] == task and r["seed"] in seeds
                        for m in r["messages"]
                    ],
                }
            ]
        }
    )


def credit(obs, ctx, report, prior=()):
    return discovery_feedback(
        obs,
        ctx,
        report,
        reviewer_provenance={
            "model": "scripted/no real provider",
            "response_ref": "scripted",
            "input_sha256": sha256_json(ctx),
            "response_sha256": sha256_json(report),
        },
        prior_keys=prior,
        min_replications=2,
    )


def trial(i, arm, pair="pair", reward=0, cost=None, feedback=None):
    return {
        "protocol_version": PROTOCOL,
        "trial_id": f"trial-{i}",
        "arm": arm,
        "service_pair_id": pair,
        "reward": reward,
        "candidate_validity": "valid",
        "feedback": feedback or {"status": "inconclusive", "discovery_keys": []},
        "cost": cost
        or {
            "native_episodes": 4,
            "api_calls": 12,
            "review_calls": 4,
            "prompt_tokens": 100,
            "completion_tokens": 20,
        },
    }


def test_four_observed_classes_and_no_causal_claim():
    obs, ctx = observations()
    assert [c["classification"] for c in obs["cells"]] == ["repaired"] * 2 + [
        "residual"
    ] * 2 + ["regression"] * 2 + ["stable"] * 2
    assert not obs["causal_service_failure_confirmed"]
    assert "NEVER_COPY_HIDDEN" not in json.dumps(ctx)
    assert all(c["repair_related"] is None for c in obs["cells"])


@pytest.mark.parametrize(
    "change", ["missing", "seed", "service", "customer", "unknown", "duplicate"]
)
def test_mismatched_or_incomplete_pairs_reject(change):
    pair, c, a, b, validity = samples()
    if change == "missing":
        b.pop()
    elif change == "duplicate":
        b.append(b[0])
    else:
        b[0] = replace(
            b[0],
            **{
                "seed": {"seed": 99},
                "service": {"service_strategy_id": "wrong"},
                "customer": {"customer_strategy_id": "wrong"},
                "unknown": {"task_success": None, "status": "uncertain"},
            }[change],
        )
    with pytest.raises(ValueError):
        paired_observations(
            a,
            b,
            pair=pair,
            customer_id=customer_strategy_id(c),
            task_ids=["1", "2", "3", "4"],
            seeds=[1, 2],
            validity=validity,
        )


@pytest.mark.parametrize("status", ["invalid", "uncertain"])
def test_actual_customer_nonvalid_has_no_reward(status):
    obs, ctx = observations()
    obs["cells"][0]["customer_validity"]["before"]["status"] = status
    assert credit(obs, ctx, supported_review(ctx))["reward"] == 0


def test_supported_replicated_novel_residual_and_regression_only():
    obs, ctx = observations()
    for task, kind in (("2", "residual"), ("3", "regression")):
        feedback = credit(obs, ctx, supported_review(ctx, task, kind))
        assert feedback["reward"] == 1
        assert (
            credit(
                obs, ctx, supported_review(ctx, task, kind), feedback["discovery_keys"]
            )["reward"]
            == 0
        )
    assert credit(obs, ctx, supported_review(ctx, seeds=(1,)))["reward"] == 0
    assert (
        discovery_feedback(
            obs, ctx, None, reviewer_provenance=None, prior_keys=[], min_replications=2
        )["status"]
        == "pending"
    )
    report = supported_review(ctx)
    report["discoveries"][0]["repair_related"] = False
    assert credit(obs, ctx, report)["reward"] == 0


@pytest.mark.parametrize(
    "change",
    ["task", "label", "hash", "endpoint", "user_only", "missing_endpoint", "schema"],
)
def test_invalid_evidence_rejects_strictly(change):
    _, ctx = observations()
    report = supported_review(ctx)
    item = report["discoveries"][0]
    if change == "task":
        item["task_id"] = "H-task"
    elif change == "label":
        item["discovery_type"] = "regression"
    elif change == "hash":
        item["evidence_refs"][0]["message_sha256"] = "forged"
    elif change == "endpoint":
        item["evidence_refs"][0]["endpoint"] = "gold"
    elif change == "user_only":
        for row in ctx["evidence"]:
            row["messages"][0]["role"] = "user"
    elif change == "missing_endpoint":
        item["evidence_refs"] = [
            r for r in item["evidence_refs"] if r["endpoint"] == "after"
        ]
    else:
        item["extra"] = "no coercion"
    with pytest.raises(ValueError):
        validate_review(report, ctx)


def test_shadow_same_service_and_no_promotion_cannot_reward():
    assert (
        service_pair(
            ServiceSkillMemoryV2(), ServiceSkillMemoryV2(), generation=0, promoted=True
        )
        is None
    )
    p = pair_fixture()
    assert (
        service_pair(
            ServiceSkillMemoryV2(),
            ServiceSkillMemoryV2.from_mapping(p["after"]),
            generation=0,
            promoted=False,
        )
        is None
    )
    obs, ctx = observations()
    obs["pair_kind"] = "shadow_repair"
    assert credit(obs, ctx, supported_review(ctx))["status"] == "shadow_diagnostic"


def test_ucb_cold_start_quota_version_reset_and_exact_restore():
    state = new_state(DEFAULT_POLICY)
    selected = []
    for i in range(10):
        choice = choose_arm(state, "pair-A", DEFAULT_POLICY)
        selected.append(choice["arm"])
        state = record_trial(state, trial(i, choice["arm"], "pair-A"))
        assert restore(serialize(state), DEFAULT_POLICY) == state
    assert selected[:5] == list(ARMS)
    assert selected[9] == "open_exploration"
    assert choose_arm(state, "pair-B", DEFAULT_POLICY)["arm"] == ARMS[0]
    assert (
        choose_arm(state, None, DEFAULT_POLICY)["frontier_status"] == "no_repair_pair"
    )
    assert sum(t["cost"]["api_calls"] for t in state["trials"]) == 120


def test_window_eviction_and_deterministic_ties():
    p = {**DEFAULT_POLICY, "window_trials": 5}
    state = new_state(p)
    for i, arm in enumerate(ARMS):
        state = record_trial(state, trial(i, arm))
    assert choose_arm(state, "pair", p)["arm"] == ARMS[0]
    state = record_trial(state, trial(5, ARMS[0]))
    state = record_trial(state, trial(6, ARMS[0]))
    assert choose_arm(state, "pair", p)["arm"] == ARMS[1]


@pytest.mark.parametrize(
    "bad",
    [
        {"beta": True},
        {"beta": float("nan")},
        {"feedback_seeds": [1]},
        {"feedback_seeds": [1, True]},
        {"max_trials_per_generation": 0},
        {"unknown": 1},
        {"open_exploration_interval": 6},
        {"min_replications": 1},
    ],
)
def test_policy_strict(bad):
    with pytest.raises(ValueError):
        validate_policy(bad)


def test_trial_idempotency_conflict_reward_cost_and_restore_validation():
    state = new_state(DEFAULT_POLICY)
    t = trial(0, ARMS[0])
    state = record_trial(state, t)
    assert record_trial(state, t) == state
    with pytest.raises(ValueError):
        record_trial(state, {**t, "cost": {"api_calls": 99}})
    with pytest.raises(ValueError):
        record_trial(state, trial(1, ARMS[1], reward=1))
    with pytest.raises(ValueError):
        record_trial(state, trial(1, ARMS[1], cost={"api_calls": -1}))
    with pytest.raises(ValueError):
        restore(state, {**DEFAULT_POLICY, "beta": 2})
    with pytest.raises(ValueError):
        restore({**state, "trials": [t, t]}, DEFAULT_POLICY)


def rc_policy():
    p = policy()
    p["repair_conditioned_bandit"] = validate_policy({})
    return p


class RCProviders(SkillProviders):
    def propose_skill_mutation(self, context):
        assert "secret-hidden" not in json.dumps(context)
        assert "repair_context" not in context
        assert "bandit_state" not in context
        return super(SkillProviders, self).propose_skill_mutation(context)

    def review_repair_discoveries(self, context):
        self.calls.append("repair_reviewer")
        assert "secret-hidden" not in json.dumps(context)
        # No invented positive finding; this mock has no repair-related replicated residual.
        return {"discoveries": []}


def test_g2_full_loop_uses_only_previous_promoted_pair_and_keeps_selection_gates(
    tmp_path,
):
    provider, native = RCProviders(), FakeRunner()
    result, _, _ = run(
        tmp_path, generations=2, provider=provider, runner=native, policy=rc_policy()
    )
    a, b = result.generations
    assert a["service_phase"]["accepted"]
    assert a["repair_service_pair"]["created_generation"] == 0
    assert provider.contexts[0]["repair_context"]["frontier_status"] == "no_repair_pair"
    assert provider.contexts[1]["repair_context"]["created_generation"] == 0
    assert b["customer_phase"]["candidates"][0]["bandit_trial"]["observations"]
    assert len(b["bandit_state"]["trials"]) == 2
    assert all(t["reward"] == 0 for t in b["bandit_state"]["trials"])
    assert list((tmp_path / "rc-bandit-trials").glob("*-final.json"))
    calls = (len(provider.calls), len(native.calls))
    resumed, _, _ = run(
        tmp_path, generations=2, provider=provider, runner=native, policy=rc_policy()
    )
    assert (len(provider.calls), len(native.calls)) == calls
    assert resumed.generations == result.generations


@pytest.mark.parametrize(
    "stage", ["rc-0-proposal", "rc-0-trial-final", "generation_complete"]
)
def test_interrupt_resume_does_not_repeat_calls_or_feedback(tmp_path, stage):
    provider, native = RCProviders(), FakeRunner()
    with pytest.raises(RuntimeError, match="interruption"):
        run(
            tmp_path / "interrupted",
            generations=2,
            provider=provider,
            runner=native,
            policy=rc_policy(),
            interrupt=stage,
        )
    result, _, _ = run(
        tmp_path / "interrupted",
        generations=2,
        provider=provider,
        runner=native,
        policy=rc_policy(),
    )
    clean, cp, cr = run(
        tmp_path / "clean",
        generations=2,
        provider=RCProviders(),
        runner=FakeRunner(),
        policy=rc_policy(),
    )
    assert len(provider.calls) == len(cp.calls)
    assert len(native.calls) == len(cr.calls)
    assert (
        result.generations[-1]["bandit_state"] == clean.generations[-1]["bandit_state"]
    )


@pytest.mark.parametrize(
    "status,static", [("invalid", True), ("uncertain", True), ("valid", False)]
)
def test_illegal_customer_trial_has_zero_reward_and_cost_is_not_ignored(
    tmp_path, status, static
):
    provider = RCProviders(status, static)
    result, _, _ = run(tmp_path, provider=provider, policy=rc_policy())
    t = result.generations[0]["bandit_state"]["trials"][0]
    assert t["reward"] == 0
    assert t["cost"]["simulated_provider_calls"] >= 2
    assert not result.generations[0]["customer_phase"]["candidates"][0]["selected"]


def test_schema_invalid_and_duplicate_consumed_trials(tmp_path):
    from evotau.evolution_candidates import EvolverSchemaError

    class Invalid(RCProviders):
        def customers(self, context, count):
            self.calls.append("invalid schema")
            raise EvolverSchemaError("malformed structure")

    result, _, _ = run(tmp_path / "schema", provider=Invalid(), policy=rc_policy())
    assert (
        result.generations[0]["bandit_state"]["trials"][0]["cost"][
            "simulated_provider_calls"
        ]
        == 1
    )

    class Duplicate(RCProviders):
        def customers(self, context, count):
            return super().customers({**context, "generation": 0}, count)

    result, _, _ = run(
        tmp_path / "duplicate", provider=Duplicate(), generations=2, policy=rc_policy()
    )
    t = result.generations[1]["bandit_state"]["trials"][-1]
    assert t["candidate_validity"] == "invalid" and "duplicate" in t["rejection"]
    assert t["cost"]["simulated_provider_calls"] == 1


def test_provider_error_keeps_failed_pull_and_cost_evidence(tmp_path):
    from evotau.budget import ProviderBudgetExceeded

    class Error(RCProviders):
        def customers(self, context, count):
            self.calls.append("denied")
            raise ProviderBudgetExceeded("scripted exhausted")

    with pytest.raises(ProviderBudgetExceeded):
        run(tmp_path, provider=Error(), policy=rc_policy())
    events = list((tmp_path / "rc-bandit-trials").glob("*-failure-*.json"))
    assert len(events) == 1
    assert (
        json.loads(events[0].read_text())["payload"]["cost"]["simulated_provider_calls"]
        == 1
    )
    assert not (tmp_path / "checkpoint.json").exists()


def test_legacy_optout_and_incompatible_restore(tmp_path):
    original, _, _ = run(
        tmp_path / "legacy", policy=policy(), provider=SkillProviders()
    )
    assert "bandit_protocol" not in original.generations[0]
    assert not (tmp_path / "legacy" / "rc-bandit-trials").exists()
    with pytest.raises(ValueError, match="Bandit protocol"):
        run(tmp_path / "legacy", policy=rc_policy(), provider=RCProviders())


def test_freeze_policy_manifest_optin_legacy_serialization_and_budget():
    models = dict.fromkeys(("agent", "customer", "evaluator", "evolver"), "openai/mock")
    args = {role: {"temperature": 0.0} for role in models}
    assert "repair_conditioned_bandit" not in freeze_v2_policy(policy(), models, args)
    assert freeze_v2_policy(rc_policy(), models, args)[
        "repair_conditioned_bandit"
    ] == validate_policy({})
    bad = rc_policy()
    del bad["customer_evolution"]
    with pytest.raises(ValueError):
        freeze_v2_policy(bad, models, args)


def test_uniform_current_failure_rc_ucb_mock_equal_total_cost():
    # Synthetic opportunity model: failure-rate favors arm 0; replicated repair
    # evidence credits arm 2 once. Same COMPLETE paired/review workload in all arms.
    from math import log, sqrt

    obs, ctx = observations()
    supported = credit(obs, ctx, supported_review(ctx))
    totals, traces = [], {}
    for mode in ("uniform", "current_failure", "rc"):
        state = new_state(DEFAULT_POLICY)
        pulls, failures = [], []
        for i in range(10):
            if mode == "uniform":
                arm = ARMS[i % 5]
            elif mode == "rc":
                arm = choose_arm(state, obs["service_pair_id"], DEFAULT_POLICY)["arm"]
            elif i < 5:
                arm = ARMS[i]
            elif "open_exploration" not in pulls[-4:]:
                arm = "open_exploration"
            else:

                def score(a, pulls=pulls, failures=failures, i=i):
                    count = pulls.count(a)
                    mean = (
                        sum(f for p, f in zip(pulls, failures, strict=True) if p == a)
                        / count
                    )
                    return mean + sqrt(log(1 + i) / (1 + count))

                arm = max(ARMS, key=score)
            pulls.append(arm)
            failures.append(float(arm == ARMS[0]))
            # Exact novelty credit only once; baselines still pay identical reviews.
            reward = int(
                mode == "rc"
                and arm == ARMS[2]
                and not any(t["reward"] for t in state["trials"])
            )
            state = record_trial(
                state,
                trial(
                    i,
                    arm,
                    obs["service_pair_id"],
                    reward=reward,
                    feedback=supported if reward else None,
                ),
            )
        traces[mode] = pulls
        totals.append(
            {
                key: sum(t["cost"][key] for t in state["trials"])
                for key in state["trials"][0]["cost"]
            }
        )
    assert totals[0] == totals[1] == totals[2]
    assert traces["current_failure"][5] == ARMS[0]
    assert traces["rc"][5] == ARMS[2]
    assert traces["rc"] != traces["uniform"]


def test_journal_trial_choices_immutable(tmp_path):
    journal = EvolutionJournal(tmp_path, "frozen")
    state = new_state(DEFAULT_POLICY)
    inputs = {"state": state, "pair": "v1"}
    choice = journal.freeze(
        "arm", inputs, lambda: choose_arm(state, "v1", DEFAULT_POLICY)
    )
    assert (
        journal.freeze("arm", inputs, lambda: pytest.fail("repeated decision"))
        == choice
    )
    with pytest.raises(ValueError):
        journal.freeze("arm", {**inputs, "pair": "v2"}, lambda: None)


def test_positive_trial_credited_once_and_does_change_ucb_priority():
    obs, ctx = observations()
    feedback = credit(obs, ctx, supported_review(ctx))
    state = new_state(DEFAULT_POLICY)
    for i, arm in enumerate(ARMS):
        state = record_trial(
            state,
            trial(
                i,
                arm,
                pair=obs["service_pair_id"],
                reward=int(i == 2),
                feedback=feedback if i == 2 else None,
            ),
        )
    assert (
        choose_arm(state, obs["service_pair_id"], DEFAULT_POLICY)["arm"]
        == "authorization_boundary"
    )
    with pytest.raises(ValueError, match="duplicate discovery"):
        record_trial(
            state,
            trial(
                99, ARMS[0], pair=obs["service_pair_id"], reward=1, feedback=feedback
            ),
        )


def test_repair_review_checks_provenance_and_novelty():
    obs, ctx = observations()
    report = supported_review(ctx)
    with pytest.raises(ValueError, match="provenance"):
        discovery_feedback(
            obs,
            ctx,
            report,
            reviewer_provenance={"model": "mock"},
            prior_keys=[],
            min_replications=2,
        )
    report["discoveries"][0]["novel"] = False
    assert credit(obs, ctx, report)["reward"] == 0


def test_v_enabled_rc_does_not_give_V_to_customer_or_ledger(tmp_path):
    provider = RCProviders()
    result, _, _ = run(tmp_path, provider=provider, policy=rc_policy(), validation=True)
    text = json.dumps(result.generations[0]["bandit_state"])
    assert "secret-hidden-4" not in text
    assert all(
        c["task_id"] != "4"
        for t in result.generations[0]["bandit_state"]["trials"]
        for c in (t.get("observations") or {}).get("cells", [])
    )
    assert not result.generations[0]["service_phase"][
        "accepted"
    ]  # unchanged small-V INCONCLUSIVE
    assert result.generations[0]["repair_service_pair"] is None
    for context in provider.contexts:
        assert "secret-hidden-4" not in json.dumps(context)


def test_frozen_candidate_budget_and_source_changes_fail_before_generation(tmp_path):
    p = rc_policy()
    p["repair_conditioned_bandit"]["max_trials_per_generation"] = 0
    with pytest.raises(ValueError):
        run(tmp_path, provider=RCProviders(), policy=p)
    p = rc_policy()
    run(tmp_path / "done", provider=RCProviders(), policy=p)
    p["repair_conditioned_bandit"]["beta"] = 2
    with pytest.raises(ValueError, match="input changed"):
        run(tmp_path / "done", provider=RCProviders(), policy=p)


def test_actual_budget_includes_denials_tokens_and_review_cost(tmp_path):
    from test_customer_skill_protocol import tasks

    from evotau.budget import RequestBudget
    from evotau.customer_diagnostic import RepairConditionedTrials

    budget = RequestBudget(1)
    budget.dispatch_external_call(
        model="scripted",
        dispatch=lambda: {"usage": {"prompt_tokens": 123, "completion_tokens": 9}},
        call_name="evotau_customer_trajectory_validator_v1",
    )
    session = RepairConditionedTrials(
        policy=DEFAULT_POLICY,
        customer_policy=policy()["customer_evolution"],
        tasks=tasks(),
        runner=FakeRunner(),
        providers=RCProviders(),
        domain_policy="native",
        root=tmp_path,
        manifest_sha="mock",
        budget=budget,
    )
    costs = session.costs()
    assert costs["api_calls"] == 1 and costs["review_calls"] == 1
    assert costs["prompt_tokens"] == 123 and costs["completion_tokens"] == 9


def test_pair_evidence_keeps_same_tasks_all_seeds_and_no_hidden_data():
    from evotau.evolution_context import build_repair_pair_evidence

    obs, ctx = observations()
    rows = {
        name: [
            {
                "task": {"task_id": r["task_id"], "user_scenario": "SECRET"},
                "seed": r["seed"],
                "trajectory_ref": "mock",
                "native_evaluation": {"task_success": False},
                "trajectory": {"messages": r["messages"]},
            }
            for r in ctx["evidence"]
            if r["endpoint"] == name
        ]
        for name in ("before", "after")
    }
    built = build_repair_pair_evidence(
        rows, obs, representative_tasks=3, case_chars=24000
    )
    selected = [
        {
            (r["task"]["task_id"], r["seed"])
            for r in built[name]
            if r["trajectory"]["messages"]
        }
        for name in ("before", "after")
    ]
    assert selected[0] == selected[1]
    assert {s for _, s in selected[0]} == {1, 2}
    assert all((t, 1) in selected[0] and (t, 2) in selected[0] for t, _ in selected[0])
    assert "SECRET" not in json.dumps(built)


def test_positive_feedback_cannot_be_relabelled_into_another_service_version():
    obs, ctx = observations()
    feedback = credit(obs, ctx, supported_review(ctx))
    with pytest.raises(ValueError, match="unverified"):
        record_trial(
            new_state(DEFAULT_POLICY),
            trial(0, ARMS[0], "other-version", reward=1, feedback=feedback),
        )


def test_evidence_of_another_episode_is_not_accepted():
    obs, ctx = observations()
    row = ctx["evidence"][0]
    rows = {
        name: [
            {
                "task": {"task_id": r["task_id"]},
                "seed": r["seed"],
                "trajectory_ref": "wrong-source",
                "trajectory": {"messages": r["messages"]},
            }
            for r in ctx["evidence"]
            if r["endpoint"] == name
        ]
        for name in ("before", "after")
    }
    assert row
    with pytest.raises(ValueError, match="different episode"):
        review_context(obs, ctx["pair"], rows, prior_discoveries=[])


def test_ledger_failure_and_final_evidence_remain_immutable(tmp_path):
    from evotau.evolution_archive import BanditTrialLedger

    ledger = BanditTrialLedger(tmp_path, "mock-manifest", ["1"])
    ledger.publish(
        "g0000-c0000",
        "failure",
        {"error_type": "TimeoutError", "cost": {"api_calls": 1}},
    )
    ledger.publish("g0000-c0000", "final", {"reward": 0, "cost": {"api_calls": 1}})
    assert len(list(ledger.root.glob("*.json"))) == 2
    with pytest.raises(FileExistsError):
        ledger.publish("g0000-c0000", "final", {"reward": 1, "cost": {"api_calls": 2}})
    with pytest.raises(ValueError):
        BanditTrialLedger(tmp_path, "other-manifest", ["1"])


def test_previous_discovery_is_not_novel_again_after_a_service_version_change():
    obs, ctx = observations()
    report = supported_review(ctx)
    first = credit(obs, ctx, report)
    changed = deepcopy(obs)
    changed["service_pair_id"] = "different-repair-signature"
    assert credit(changed, ctx, report, first["discovery_keys"])["reward"] == 0
