from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from evotau.archive import FailureArchive
from evotau.budget import ProviderBudgetExceeded, RequestBudget
from evotau.lifecycle import (
    ServiceTransitionEpisodeRunner,
    TwoGenerationSmoke,
    _append_prepared_archive,
    _prepared_archive_payload,
)
from evotau.records import (
    EpisodeRecord,
    EpisodeStatus,
    EvidenceRef,
    FailureRecord,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.service_evolution import RepairAudit, RepairProposal
from evotau.service_transition import GatedServiceTransition
from evotau.strategies import CustomerStrategy, ServiceRule, ServiceStrategy


def _episode(
    name: str,
    *,
    task: str,
    seed: int,
    customer: CustomerStrategy | None,
    service: ServiceStrategy,
    success: bool,
    violation: bool = False,
    native: bool = False,
) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=name,
        task_id=task,
        seed=seed,
        customer_strategy_id=customer_strategy_id(customer),
        service_strategy_id=service_strategy_id(service),
        status=EpisodeStatus.COMPLETE,
        task_success=success,
        customer_valid=True,
        strategy_applicable=not native,
        customer_strategy_adherent=None if native else True,
        policy_violation=violation,
        invalid_repeated_write_calls=0,
        policy_rule_id="retail.policy:explicit_confirmation" if violation else None,
        mistake_type="missing_explicit_confirmation" if violation else None,
        workflow_stage="pre_write" if violation else None,
        evidence=(EvidenceRef(2, "tool", "write occurred before confirmation"),) if violation else (),
        tool_calls=4,
    )


def _fixture(*, remaining_episodes: int = 10, audit_approved: bool = True):
    customer = CustomerStrategy(challenge_style="ask_reason", challenge_budget=1)
    incumbent = ServiceStrategy()
    customer_id = customer_strategy_id(customer)
    incumbent_id = service_strategy_id(incumbent)
    source = _episode(
        "source-failure", task="E", seed=3, customer=customer, service=incumbent,
        success=False, violation=True,
    )
    reproduction = _episode(
        "source-failure-replay", task="E", seed=10_003, customer=customer,
        service=incumbent, success=False, violation=True,
    )
    failure = FailureRecord.verify(
        source, reproduction_episode=reproduction, generation=0,
        verifier="human:target-audit", reproduction_verifier="human:target-replay-audit",
    )
    historical = _episode(
        "source-pass", task="E", seed=4, customer=customer, service=incumbent,
        success=True,
    )
    calls: list[dict] = []

    def runner(*, task_id, seed, customer, service, panel_name):
        calls.append({
            "task_id": task_id, "seed": seed, "customer": customer,
            "service": service, "panel_name": panel_name,
        })
        panel = "target" if "target" in panel_name else "other"
        baseline = service_strategy_id(service) == incumbent_id
        violation = panel == "target" and baseline
        return _episode(
            panel_name, task=task_id, seed=seed, customer=customer, service=service,
            success=not violation, violation=violation, native=customer is None,
        )

    episode_runner = ServiceTransitionEpisodeRunner(
        runner, remaining_episodes, {customer_id: customer}, (source, historical),
        service_policy_text="Fixed retail policy: obtain explicit confirmation before writing.",
    )
    proposals = []

    def proposal_provider(repair_input):
        target = repair_input.target_failure
        proposals.append((repair_input.generation, target.failure_id, repair_input.current_service))
        assert repair_input.target_episode == source
        assert repair_input.fixed_policy_text.startswith("Fixed retail policy")
        rule = ServiceRule(
            "confirmation-guard", target.policy_ref,
            "before a database write",
            "summarize all action details and get explicit yes confirmation before writing",
            target.evidence_refs,
        )
        return RepairProposal(
            target.failure_id, rule,
            "On the same eligible write request, the agent confirms all details before writing.",
        )

    def audit_provider(proposal, repair_input):
        target = repair_input.target_failure
        return RepairAudit(
            audit_approved,
            "human:repair-audit",
            frozenset({target.policy_ref}) if audit_approved else frozenset(),
            rationale="The repair preserves permissions and restates the fixed confirmation policy.",
        )

    transition = GatedServiceTransition(
        "E", "V", 10, incumbent, proposal_provider, audit_provider,
        token_counter=lambda text: len(text.split()),
    )
    return customer, incumbent, failure, historical, calls, proposals, episode_runner, transition


def test_gated_transition_executes_complete_paired_gate_and_accepts_only_candidate():
    customer, incumbent, failure, _historical, calls, proposals, episode_runner, transition = _fixture()
    candidate, report, note = transition(
        0, customer, incumbent, (failure,), episode_runner, None,
    )

    assert report is not None and report.accepted and not report.inconclusive
    assert candidate == report.candidate_strategy
    assert report.target_failure_id == failure.failure_id
    assert report.initial_s0_episode_refs == (("clean-1", "service-g0-clean-incumbent"),)
    assert len(report.unit_episode_refs) == 5
    assert report.unit_episode_refs[0] == (
        "target-1", "service-g0-target-1-incumbent", "service-g0-target-1-candidate",
    )
    assert len(calls) == 10
    assert len(proposals) == 1 and proposals[0][1] == failure.failure_id
    assert note.startswith("accepted:")
    target_calls = [call for call in calls if "target" in call["panel_name"]]
    assert len(target_calls) == 4
    for index in (1, 2):
        pair = [call for call in target_calls if f"target-{index}-" in call["panel_name"]]
        assert len(pair) == 2
        assert pair[0]["seed"] == pair[1]["seed"]
        assert pair[0]["customer"] == pair[1]["customer"]
        assert pair[0]["service"] == incumbent
        assert pair[1]["service"] == candidate
    assert any(call["task_id"] == "V" and call["customer"] == customer for call in calls)


def test_no_historical_replay_ablation_skips_history_and_records_gate_condition():
    customer, incumbent, failure, _historical, calls, proposals, episode_runner, transition = _fixture()
    source = episode_runner.find_failure_episode(failure)
    episode_runner.episode_history = (source,)
    transition = replace(transition, include_historical_replay=False)

    candidate, report, note = transition(
        0, customer, incumbent, (failure,), episode_runner, None,
    )

    assert candidate != incumbent
    assert report is not None and report.accepted
    assert report.historical_replay_included is False
    assert report.to_dict()["historical_replay_included"] is False
    assert all(unit[0] != "historical-1" for unit in report.unit_episode_refs)
    assert len(report.unit_episode_refs) == 4
    assert len(calls) == 8
    assert not any("historical" in call["panel_name"] for call in calls)
    assert sorted({call["seed"] for call in calls}) == [20_010, 20_011, 20_012, 20_013]
    assert len(proposals) == 1
    assert proposals[0][1] == failure.failure_id
    assert note.startswith("accepted:")


def test_gated_transition_pairs_clean_and_validation_units_across_pilot_panels():
    customer, incumbent, failure, _historical, calls, _proposals, episode_runner, transition = _fixture(
        remaining_episodes=14,
    )
    transition = replace(
        transition,
        evolution_task_ids=("E", "E2"),
        validation_task_ids=("V", "V2"),
    )

    candidate, report, note = transition(0, customer, incumbent, (failure,), episode_runner, None)

    assert candidate != incumbent
    assert report is not None and report.accepted
    assert len(report.unit_episode_refs) == 7
    clean_tasks = {
        call["task_id"] for call in calls if "-clean-" in call["panel_name"]
    }
    validation_tasks = {
        call["task_id"] for call in calls if "validation-" in call["panel_name"]
    }
    assert clean_tasks == {"E", "E2"}
    assert validation_tasks == {"V", "V2"}
    assert len(calls) == 14
    assert note.startswith("accepted:")


def test_gated_transition_preflights_episode_and_request_budgets_before_proposal():
    customer, incumbent, failure, _historical, calls, proposals, runner, transition = _fixture(
        remaining_episodes=9,
    )
    result, report, note = transition(0, customer, incumbent, (failure,), runner, None)
    assert result == incumbent and report is None
    assert "inconclusive" in note and "10 episode slots" in note
    assert not calls and not proposals

    customer, incumbent, failure, _historical, calls, proposals, runner, transition = _fixture()
    budget = RequestBudget(11)
    result, report, note = transition(0, customer, incumbent, (failure,), runner, budget)
    assert result == incumbent and report is None
    assert "inconclusive" in note and "12 provider attempts" in note
    assert not calls and not proposals

    customer, _initial, failure, historical, calls, proposals, runner, transition = _fixture()
    current = ServiceStrategy((ServiceRule(
        "earlier-repair", failure.policy_ref, "before a refund", "verify the refund scope",
        failure.evidence_refs,
    ),))
    runner.episode_history = (
        runner.find_failure_episode(failure),
        replace(historical, service_strategy_id=service_strategy_id(current)),
    )
    runner.remaining_episodes = 10
    result, report, note = transition(1, customer, current, (failure,), runner, None)
    assert result == current and report is None
    assert "11 episode slots" in note
    assert not calls and not proposals


def test_gated_transition_fails_closed_without_historical_success_or_audit_approval():
    customer, incumbent, failure, _historical, calls, proposals, runner, transition = _fixture()
    runner.episode_history = (runner.find_failure_episode(failure),)
    result, report, note = transition(0, customer, incumbent, (failure,), runner, None)
    assert result == incumbent and report is None
    assert "incumbent-passing" in note
    assert not calls and not proposals

    customer, incumbent, failure, _historical, calls, _proposals, runner, transition = _fixture(
        audit_approved=False,
    )
    result, report, note = transition(0, customer, incumbent, (failure,), runner, None)
    assert result == incumbent and report is not None and not report.accepted
    assert not report.inconclusive
    assert "static" in note
    assert not calls


def test_request_budget_exhaustion_during_gate_records_inconclusive_report():
    customer, incumbent, failure, _historical, _calls, _proposals, runner, transition = _fixture()
    original_runner = runner.episode_runner

    def exhaust_on_second_target_candidate(**kwargs):
        if "target-2-candidate" in kwargs["panel_name"]:
            raise ProviderBudgetExceeded("provider request denied before dispatch")
        return original_runner(**kwargs)

    runner.episode_runner = exhaust_on_second_target_candidate
    result, report, note = transition(0, customer, incumbent, (failure,), runner, None)
    assert result == incumbent and report is not None
    assert report.inconclusive and not report.accepted
    assert len(report.unit_results) == 1
    assert report.partial_episode_refs == ((
        "target-2:incumbent", "service-g0-target-2-incumbent",
    ),)
    assert "inconclusive" in note


def test_provider_cost_preflight_stops_before_dispatch_and_journals_target_replay(tmp_path):
    customer, incumbent, failure, _historical, calls, proposals, runner, transition = _fixture()
    remaining = {"attempts": 12}

    class Budget:
        def snapshot(self):
            return SimpleNamespace(remaining=remaining["attempts"])

    original_proposal = transition.proposal_provider
    original_audit = transition.audit_provider

    def proposal_provider(*args):
        remaining["attempts"] -= 2
        return original_proposal(*args)

    def audit_provider(*args):
        remaining["attempts"] -= 1
        return original_audit(*args)

    transition = replace(
        transition, proposal_provider=proposal_provider, audit_provider=audit_provider,
    )
    result, report, note = transition(0, customer, incumbent, (failure,), runner, Budget())
    assert result == incumbent and report is not None and report.inconclusive
    assert "no gate episode was dispatched" in note
    assert not calls and len(proposals) == 1

    customer, incumbent, failure, _historical, _calls, _proposals, runner, transition = _fixture()
    candidate, report, _note = transition(0, customer, incumbent, (failure,), runner, None)
    assert candidate != incumbent and report is not None and report.accepted
    archive = FailureArchive(tmp_path / "replays.sqlite")
    payload = _prepared_archive_payload(
        generation=0, old_customer=customer, proposals=(), failures=(failure,),
        old_service=incumbent, new_service=candidate, service_gate=report,
    )
    _append_prepared_archive(archive, payload)
    _append_prepared_archive(archive, payload)
    assert archive.active_replay_coverage()["coverage_rate"] == 1.0


def test_concrete_transition_completes_through_controller_and_archives_replay(tmp_path):
    customer = CustomerStrategy()
    service = ServiceStrategy()
    archive = FailureArchive(tmp_path / "controller-archive.sqlite")

    def runner(*, task_id, seed, customer, service, panel_name):
        native = customer is None
        discovery_target = (
            panel_name in {"discovery", "confirmation"}
            and customer_strategy_id(customer) == customer_strategy_id(CustomerStrategy())
            and not service.rules
        )
        gate_target = "target-" in panel_name and not service.rules
        violation = discovery_target or gate_target
        return EpisodeRecord(
            episode_id=f"{panel_name}:{task_id}:{seed}:{customer_strategy_id(customer)}",
            task_id=task_id,
            seed=seed,
            customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service),
            status=EpisodeStatus.COMPLETE,
            task_success=not violation,
            customer_valid=True,
            strategy_applicable=not native,
            customer_strategy_adherent=None if native else True,
            policy_violation=violation,
            invalid_repeated_write_calls=0,
            policy_rule_id="retail.policy:explicit_confirmation" if violation else None,
            mistake_type="missing_explicit_confirmation" if violation else None,
            workflow_stage="pre_write" if violation else None,
            evidence=(EvidenceRef(2, "tool", "write preceded confirmation"),) if violation else (),
            tool_calls=4,
        )

    def proposal_provider(repair_input):
        failure = repair_input.target_failure
        return RepairProposal(
            failure.failure_id,
            ServiceRule(
                "controller-confirmation",
                failure.policy_ref,
                "before a write",
                "confirm the full write details before calling the tool",
                failure.evidence_refs,
            ),
            "The write occurs only after the customer confirms its complete scope.",
        )

    def audit_provider(_proposal, repair_input):
        failure = repair_input.target_failure
        return RepairAudit(
            True,
            "human:controller-policy-review",
            frozenset({failure.policy_ref}),
            rationale="This rule enforces the fixed confirmation policy without expanding permission.",
        )

    transition = GatedServiceTransition(
        "E", "V", 12, service, proposal_provider, audit_provider,
        token_counter=lambda text: len(text.split()),
    )
    controller = TwoGenerationSmoke(
        manifest={"max_episodes": 23},
        checkpoint_path=str(tmp_path / "controller-checkpoint.json"),
        runner=runner,
        task_ids=("E", "V"),
        seed=12,
        failure_archive=archive,
    )
    commits = controller.run(
        customer,
        service,
        failure_verifier=lambda _episode: "human:service-failure-review",
        service_transition=transition,
    )

    assert len(commits) == 2
    first = commits[0].decision_record
    assert commits[0].service_evolved
    assert first is not None
    assert first["service"]["gate"]["accepted"] is True
    assert len(first["service"]["gate"]["unit_episode_refs"]) == 5
    assert first["active_replay_coverage"]["coverage_rate"] == 1.0
    assert archive.active_replay_coverage()["coverage_rate"] == 1.0
