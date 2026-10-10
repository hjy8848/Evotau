"""Synthetic contract/control-flow evidence, never live effectiveness measurements."""

import json
from dataclasses import replace

import pytest
from test_customer_skill_protocol import SkillProviders, policy, skill
from test_repair_conditioned_bandit import (
    credit,
    observations,
    pair_fixture,
    supported_review,
)
from test_skill_evolution_v2 import FakeRunner, bind_evidence, mutation, record, run

from evotau.customer_skills import CustomerSkill
from evotau.evolution_archive import BanditTrialLedger
from evotau.open_repair_search import PROTOCOL
from evotau.open_repair_search import validate_policy as open_policy
from evotau.records import EpisodeStatus, customer_strategy_id, service_strategy_id
from evotau.repair_discovery_archive import RepairDiscoveryArchive
from evotau.service_skills import ServiceSkillMemoryV2, apply_v2_mutation
from evotau.targeted_repair_gate import evaluate_target
from evotau.targeted_repair_gate import validate_policy as target_policy


def settings(target=False):
    p = policy()
    p["open_repair_search"] = open_policy({})
    p["discovery_handoff"] = {"protocol_version": "repair_discovery_handoff_v1"}
    if target:
        p["targeted_repair"] = target_policy({})
        p["statistical_gate"].update(
            method="finite_panel_paired",
            adaptive_validation=True,
            risk_scope="observed_panel",
            enabled=True,
        )
        p["evaluation"].update(gate_seeds=[1, 2], repair_seeds=[1, 2])
    return p


def archive_fixture(tmp_path):
    obs, ctx = observations()
    report = supported_review(ctx)
    f = credit(obs, ctx, report)
    c = CustomerSkill.from_mapping(skill())
    trial = {
        "trial_id": "g0001-c0000",
        "generation": 1,
        "candidate_validity": "valid",
        "candidate_id": c.candidate_id,
        "procedure_id": c.procedure_id,
        "service_pair_id": obs["service_pair_id"],
        "observations": obs,
        "feedback": f,
        "reward": 1,
        "cost": {"api_calls": 0},
        "review_context": ctx,
    }
    ledger = BanditTrialLedger(tmp_path, "manifest", ("1", "2", "3", "4"))
    ledger.publish(trial["trial_id"], "final", trial)
    arc = RepairDiscoveryArchive(tmp_path, "manifest", ("1", "2", "3", "4"))
    kwargs = {
        "selected": False,
        "trial_ref": ledger.root / (trial["trial_id"] + "-final.json"),
        "customer_policy": {},
        "evidence_rows": [],
        "min_replications": 2,
    }
    return arc, trial, ctx, kwargs


def test_unselected_discovery_index_resume_and_conflict(tmp_path):
    arc, trial, ctx, kw = archive_fixture(tmp_path)
    docs = arc.publish(trial, skill(), pair_fixture(), ctx, **kw)
    assert len(docs) == 1 and docs[0]["selected_as_incumbent"] is False
    assert arc.publish(trial, skill(), pair_fixture(), ctx, **kw) == docs
    restored = RepairDiscoveryArchive(tmp_path, "manifest", ("1", "2", "3", "4"))
    assert len(list(restored.root.glob("discovery-*.json"))) == 1
    with pytest.raises(FileExistsError):
        arc.publish(trial, skill(), pair_fixture(), ctx, **{**kw, "selected": True})
    arc.event(docs[0]["discovery_id"], 1, "g0001-direct-0", {"deployed": False})


@pytest.mark.parametrize(
    "change",
    ["invalid", "forged_reward", "review_hash", "pair", "procedure", "evidence"],
)
def test_archive_rejects_unsupported_positive_claims(tmp_path, change):
    arc, trial, ctx, kw = archive_fixture(tmp_path)
    pair = pair_fixture()
    if change == "invalid":
        trial["candidate_validity"] = "invalid"
    elif change == "forged_reward":
        trial["feedback"]["discovery_keys"] = ["fake"]
    elif change == "review_hash":
        trial["feedback"]["review"]["discoveries"][0]["mechanism"] = "made up"
    elif change == "pair":
        pair["kind"] = "shadow_repair"
    elif change == "procedure":
        trial["procedure_id"] = "fake"
    else:
        ctx["evidence"][0]["messages"][0]["content"] = "changed"
    with pytest.raises(ValueError):
        arc.publish(trial, skill(), pair, ctx, **kw)


def target_fixture():
    c = CustomerSkill.from_mapping(skill()).compile()
    before, after = (
        ServiceSkillMemoryV2(),
        apply_v2_mutation(ServiceSkillMemoryV2(), mutation(), next_skill_id_number=1),
    )
    d = {
        "discovery_id": "a" * 64,
        "after_service_id": service_strategy_id(before),
        "compiled_customer": c.to_dict(),
        "discovery": {"task_id": "1", "seeds": [1, 2]},
    }
    old, new = [], []
    for seed in (1, 2):
        old.append(
            replace(
                record("1", False, seed),
                customer_strategy_id=customer_strategy_id(c),
                service_strategy_id=service_strategy_id(before),
            )
        )
        new.append(
            replace(
                record("1", True, seed),
                customer_strategy_id=customer_strategy_id(c),
                service_strategy_id=service_strategy_id(after),
            )
        )
    return old, new, d, before, after


def test_target_requires_native_improvement_not_reviewer():
    a, b, d, left, right = target_fixture()
    assert evaluate_target(a, b, d, left, right, {})["verdict"] == "ACCEPTED"
    b = [replace(r, task_success=False) for r in b]
    assert evaluate_target(a, b, d, left, right, {})["verdict"] == "REJECTED"


@pytest.mark.parametrize(
    "field,value",
    [
        ("hard_policy_protocol_violations", 1),
        ("termination_reason", "max_steps"),
        ("task_success", None),
    ],
)
def test_target_safety_or_unknown(field, value):
    a, b, d, left, right = target_fixture()
    b[0] = replace(
        b[0],
        **{field: value},
        **({"status": EpisodeStatus.UNCERTAIN} if value is None else {}),
    )
    assert evaluate_target(a, b, d, left, right, {})["verdict"] in (
        "REJECTED",
        "INCONCLUSIVE",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 9),
        ("task_id", "H"),
        ("customer_strategy_id", "other"),
        ("service_strategy_id", "other"),
    ],
)
def test_target_identity_strict(field, value):
    a, b, d, left, right = target_fixture()
    b[0] = replace(b[0], **{field: value})
    with pytest.raises(ValueError):
        evaluate_target(a, b, d, left, right, {})


class OpenProviders(SkillProviders):
    def review_repair_discoveries(self, context):
        self.calls.append("repair_reviewer")
        assert "secret-hidden" not in json.dumps(context)
        if any(
            c["task_id"] == "2" and c["classification"] == "residual"
            for c in context["observations"]["cells"]
        ):
            return supported_review(context)
        return {"discoveries": []}

    def propose_skill_mutation(self, context):
        self.calls.append("mutate")
        assert "secret-hidden" not in json.dumps(context)
        if context.get("discovery_repair_targets"):
            value = mutation(
                "Repair the targeted observable condition.", family="targeted"
            )
            value["evidence_task_ids"] = ["2"]
        elif context["generation"] == 0:
            value = mutation()
        else:
            value = mutation(operation="no_op")
        if context.get("discovery_repair_targets"):
            return bind_evidence(
                value,
                {
                    **context,
                    "task_interactions": [
                        r
                        for r in context["task_interactions"]
                        if r["native_evaluation"]["task_success"] is False
                    ],
                },
            )
        return bind_evidence(value, context)


class DiscoveryRunner(FakeRunner):
    def __call__(self, **kwargs):
        r = super().__call__(**kwargs)
        c = kwargs["customer"]
        challenge = c is not None and "dependency clarification 1" in c.text
        n = len(kwargs["service"].skills)
        task = kwargs["task_id"]
        ok = (
            n > 0
            if task == "1"
            else (not challenge or n > 1)
            if task == "2"
            else (challenge or n > 1)
            if task == "3"
            else True
        )
        r = replace(r, task_success=ok, native_reward=float(ok))
        self.cache[
            (
                task,
                kwargs["seed"],
                customer_strategy_id(c),
                service_strategy_id(kwargs["service"]),
            )
        ] = r
        return r


def test_real_orchestrator_g2_unselected_discovery_handoff_gate_archive_resume(
    tmp_path,
):
    p, native = OpenProviders(), DiscoveryRunner()
    result, _, _ = run(
        tmp_path,
        provider=p,
        runner=native,
        policy=settings(True),
        validation=True,
        generations=2,
    )
    first, second = result.generations
    assert first["service_phase"]["accepted"]
    row = second["customer_phase"]["candidates"][0]
    assert row["selected"] is False and row["bandit_trial"]["reward"] == 1
    assert second["service_phase"]["accepted"]
    discovery = json.loads(
        next((tmp_path / "repair-discoveries").glob("discovery-*.json")).read_text()
    )["payload"]
    assert discovery["selected_as_incumbent"] is False
    assert all("operator_family_hint" not in c for c in p.contexts)
    assert "repair_pair" not in p.contexts[0]["open_repair_search"]
    assert p.contexts[1]["open_repair_search"]["repair_pair"]["created_generation"] == 0
    calls = (len(p.calls), len(native.calls))
    resumed, _, _ = run(
        tmp_path,
        provider=p,
        runner=native,
        policy=settings(True),
        validation=True,
        generations=2,
    )
    assert resumed.generations == result.generations
    assert calls == (len(p.calls), len(native.calls))


def test_open_search_never_calls_fixed_arm_allocator(tmp_path, monkeypatch):
    import evotau.repair_conditioned_bandit as bandit

    monkeypatch.setattr(
        bandit,
        "choose_arm",
        lambda *args: (_ for _ in ()).throw(AssertionError("fixed arms called")),
    )
    result, p, _ = run(
        tmp_path, provider=OpenProviders(), policy=settings(), generations=2
    )
    assert result.generations[-1]["bandit_protocol"] == PROTOCOL
    assert all(
        t["arm"] is None for t in result.generations[-1]["bandit_state"]["trials"]
    )
    assert all("operator_family_hint" not in c for c in p.contexts)


def test_free_search_has_no_repair_condition_input(tmp_path):
    p = settings()
    p["open_repair_search"]["mode"] = "free_search"
    result, provider, _ = run(
        tmp_path, provider=OpenProviders(), policy=p, generations=2
    )
    assert all("repair_pair" not in c["open_repair_search"] for c in provider.contexts)
    assert result.generations[-1]["bandit_protocol"] == PROTOCOL


@pytest.mark.parametrize("mode", ["unknown", "disclosure_timing", "open_exploration"])
def test_open_policy_has_no_closed_directions(mode):
    with pytest.raises(ValueError):
        open_policy({"mode": mode})


def test_target_requires_independent_v(tmp_path):
    with pytest.raises(ValueError, match="independent V"):
        run(tmp_path, policy=settings(True), provider=OpenProviders())


@pytest.mark.parametrize(
    "stop",
    [
        "rc-0-trial-final",
        "customer_selection",
        "service_selection",
        "generation_complete",
    ],
)
def test_new_protocol_interrupt_replay_has_no_extra_requests(tmp_path, stop):
    p, native = OpenProviders(), DiscoveryRunner()
    with pytest.raises(RuntimeError, match="interruption"):
        run(
            tmp_path / "interrupted",
            provider=p,
            runner=native,
            policy=settings(True),
            validation=True,
            generations=2,
            interrupt=stop,
        )
    result, _, _ = run(
        tmp_path / "interrupted",
        provider=p,
        runner=native,
        policy=settings(True),
        validation=True,
        generations=2,
    )
    clean, cp, cr = run(
        tmp_path / "clean",
        provider=OpenProviders(),
        runner=DiscoveryRunner(),
        policy=settings(True),
        validation=True,
        generations=2,
    )
    assert len(p.calls) == len(cp.calls) and len(native.calls) == len(cr.calls)
    assert (
        result.generations[-1]["bandit_state"] == clean.generations[-1]["bandit_state"]
    )


@pytest.mark.parametrize("unsafe", ["E", "V"])
def test_target_fix_cannot_override_protected_regression(tmp_path, unsafe):
    class BadRunner(DiscoveryRunner):
        def __call__(self, **kwargs):
            r = super().__call__(**kwargs)
            if len(kwargs["service"].skills) > 1 and kwargs["task_id"] == (
                "3" if unsafe == "E" else "4"
            ):
                return replace(
                    r,
                    task_success=False,
                    native_reward=0.0,
                    hard_policy_protocol_violations=1 if unsafe == "E" else 0,
                )
            return r

    result, _, _ = run(
        tmp_path,
        provider=OpenProviders(),
        runner=BadRunner(),
        policy=settings(True),
        validation=True,
        generations=2,
    )
    assert result.generations[0]["service_phase"]["accepted"]
    assert not result.generations[1]["service_phase"]["accepted"]
    assert len(result.service.skills) == 1


def test_open_prompt_no_fixed_category_anchors():
    from types import SimpleNamespace

    from evotau.customer_evolution import CUSTOMER_SKILL_PROMPT
    from evotau.evolution_candidates import V2Providers

    p = V2Providers(SimpleNamespace())
    p.configure_customer(policy())
    captured = []
    p.call = lambda prompt, context, name: (
        captured.append((prompt, context, name)) or {"candidates": [skill()]}
    )
    p.customers(
        {"task_interactions": [], "open_repair_search": {"protocol_version": PROTOCOL}},
        1,
    )
    prompt, _, name = captured[-1]
    assert "Explore truthful information disclosure timing" not in prompt
    assert "operator_family_hint" not in prompt and "ARMS" not in prompt
    assert "2–5" in prompt and name == "evotau_customer_open_repair_search_v1"
    p.customers({"task_interactions": []}, 1)
    assert captured[-1][0] == CUSTOMER_SKILL_PROMPT


def test_comparison_templates_share_models_tasks_budget_and_seeds():
    from pathlib import Path

    from evotau.alternating_manifest import AlternatingManifest
    from evotau.phase0 import load_config

    manifests = [
        AlternatingManifest.from_mapping(
            load_config(Path("configs") / f"airline-{name}-v1-e10-offline.yaml")
        )
        for name in ("free-search", "rc-bandit", "open-rc")
    ]
    a = manifests[0]
    for b in manifests[1:]:
        assert b.role_models == a.role_models and b.role_model_args == a.role_model_args
        assert b.evolution_task_ids == a.evolution_task_ids
        assert b.customer_candidates == a.customer_candidates == 5
        assert (
            not b.real_provider_enabled and not b.run_heldout and not b.run_validation
        )
    assert len({m.output_path for m in manifests}) == 3
    targeted = AlternatingManifest.from_mapping(
        load_config("configs/airline-open-rc-targeted-v1-e10-v20-offline.yaml")
    )
    assert (
        targeted.run_validation
        and not targeted.run_heldout
        and not targeted.real_provider_enabled
    )


def test_open_and_bandit_config_cannot_silently_mix():
    from evotau.repair_conditioned_bandit import DEFAULT_POLICY
    from evotau.skill_evolution_config import freeze_v2_policy

    models = dict.fromkeys(("agent", "customer", "evaluator", "evolver"), "openai/mock")
    args = {k: {"temperature": 0.0} for k in models}
    p = settings()
    assert (
        freeze_v2_policy(p, models, args)["open_repair_search"]["protocol_version"]
        == PROTOCOL
    )
    p["repair_conditioned_bandit"] = DEFAULT_POLICY
    with pytest.raises(ValueError, match="exclusive"):
        freeze_v2_policy(p, models, args)


def test_no_pair_or_invalid_candidate_has_no_discovery(tmp_path):
    provider = OpenProviders(status="uncertain")
    result, _, _ = run(tmp_path, provider=provider, policy=settings(), generations=1)
    row = result.generations[0]["customer_phase"]["candidates"][0]
    assert row["bandit_trial"]["reward"] == 0
    assert not list((tmp_path / "repair-discoveries").glob("discovery-*.json"))


def test_open_trial_budget_restore_identity_and_no_repeated_credit(tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace

    from evotau.customer_skills import validate_customer_policy
    from evotau.open_repair_search import OpenRepairTrials

    search = OpenRepairTrials(
        policy={"max_trials_per_generation": 1},
        customer_policy=validate_customer_policy({}),
        tasks={"1": SimpleNamespace(id="1")},
        runner=FakeRunner(),
        providers=OpenProviders(),
        domain_policy="public",
        root=tmp_path,
        manifest_sha="manifest",
    )
    callback = lambda name, inputs, cb: cb()
    ctx = search.begin(0, 0, {"task_interactions": []}, None, callback)
    assert (
        "operator_family_hint" not in ctx
        and ctx["open_repair_search"]["frontier_status"] == "ordinary_open_exploration"
    )
    with pytest.raises(ValueError, match="budget"):
        search.begin(0, 1, {}, None, callback)
    bad = deepcopy(search.state)
    bad["protocol_version"] = "repair_conditioned_bandit_v1"
    with pytest.raises(ValueError, match="identity"):
        search.restore_state(bad)


def test_discovery_archive_detects_changed_original_trial(tmp_path):
    arc, trial, ctx, kw = archive_fixture(tmp_path)
    arc.publish(trial, skill(), pair_fixture(), ctx, **kw)
    source = kw["trial_ref"]
    value = json.loads(source.read_text())
    value["payload"]["cost"]["api_calls"] = 999
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source trial changed"):
        RepairDiscoveryArchive(tmp_path, "manifest", ("1", "2", "3", "4"))
