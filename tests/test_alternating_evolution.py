from __future__ import annotations

from dataclasses import dataclass

from evotau.alternating import (
    propose_fresh_customer_challenge,
    run_alternating_evolution,
    run_final_endpoint_evaluation,
)
from evotau.records import EpisodeRecord, EpisodeStatus
from evotau.strategies import PromptStrategy


@dataclass
class FakeTask:
    id: str
    description: str
    user_scenario: str
    user_tools: tuple[str, ...] = ()
    reference_actions: tuple[str, ...] = ("hidden gold",)
    evaluation_criteria: tuple[str, ...] = ("hidden evaluator details",)
    db_state: str = "hidden backend state"


class FakeRunner:
    service_policy_text = "Fixed benchmark policy text."

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.trajectories: dict[str, dict] = {}

    def __call__(self, *, task_id, seed, customer, service, panel_name):
        success = task_id == "v" or service.text.startswith(("Verify scope", "service "))
        episode_id = f"episode-{len(self.calls)}"
        record = EpisodeRecord(
            episode_id=episode_id,
            task_id=task_id,
            seed=seed,
            customer_strategy_id=(
                "native" if customer is None else str(hash(customer.text))
            ),
            service_strategy_id=str(hash(service.text)),
            status=EpisodeStatus.COMPLETE,
            task_success=success,
            native_reward=1.0 if success else 0.0,
            termination_reason="user_stop",
            trajectory_ref=f"{episode_id}.json",
            raw_review={
                "native_review": {
                    "agent_error": not success,
                    "summary": "resolved" if success else "service failed",
                },
            },
        )
        self.calls.append({
            "task_id": task_id,
            "seed": seed,
            "customer": customer,
            "service": service,
            "panel_name": panel_name,
        })
        self.trajectories[record.trajectory_ref] = {
            "messages": [
                {"role": "user", "content": "I need help with this request."},
                {"role": "assistant", "content": "Checking the account."},
                {"role": "tool", "name": "lookup_order", "content": "order found"},
                {"role": "assistant", "content": "Done." if success else "I cannot help."},
            ],
            "termination_reason": "user_stop",
            "reward_info": {"should_not_be_forwarded": True},
        }
        return record

    def load_trajectory(self, episode):
        return self.trajectories[episode.trajectory_ref]


def test_one_generation_runs_customer_then_service_and_records_strategy_changes(tmp_path) -> None:
    hidden_scenario = "HIDDEN SCENARIO: asks for a refund if the delivery is late."
    tasks = {
        "e": FakeTask("e", "Evolution task", hidden_scenario),
        "v": FakeTask("v", "Clean task", "Customer requests a normal order lookup."),
    }
    runner = FakeRunner()
    seen_customer_contexts = []
    seen_service_contexts = []

    def customer_evolver(context, count):
        seen_customer_contexts.append(context)
        assert count == 2
        assert context["current_service_strategy"] == "initial service"
        assert context["task_interactions"][0]["native_evaluation"]["task_success"] is False
        assert context["task_interactions"][0]["tool_results"][0]["name"] == "lookup_order"
        assert context["task_interactions"][0]["task"]["user_scenario"] == hidden_scenario
        assert "reference_actions" not in context["task_interactions"][0]["task"]
        assert "evaluation_criteria" not in context["task_interactions"][0]["task"]
        assert "should_not_be_forwarded" not in str(context)
        return ["Keep negotiating for the requested outcome.", "Ask a clarifying question first."]

    def customer_judge(context):
        assert context["service_strategy"] == "initial service"
        return {"choice": 0, "reason": "Candidate followed the scenario and exposed the current failure."}

    def service_evolver(context):
        seen_service_contexts.append(context)
        assert context["customer_strategy"] == "Keep negotiating for the requested outcome."
        assert context["service_policy"] == runner.service_policy_text
        assert context["task_interactions"][0]["native_evaluation"]["task_success"] is False
        assert "user_scenario" not in context["task_interactions"][0]["task"]
        assert hidden_scenario not in str(context)
        assert "lookup_order" in str(context["task_interactions"][0]["trajectory"])
        return {"analysis": "The service missed the customer's goal.", "strategy": "Verify scope, then complete eligible requests."}

    def service_judge(context):
        assert context["old_service_strategy"] == "initial service"
        assert context["proposed_service_strategy"].startswith("Verify scope")
        assert "user_scenario" not in context["old_service_episodes"][0]["task"]
        assert "user_scenario" not in context["proposed_service_episodes"][0]["task"]
        assert hidden_scenario not in str(context)
        return {"improved": True, "reason": "The real challenge now succeeds."}

    result = run_alternating_evolution(
        tasks=tasks,
        evolution_task_ids=("e",),
        validation_task_ids=("v",),
        seed=1,
        generations=1,
        customer_candidate_count=2,
        clean_panel_size=1,
        initial_customer=PromptStrategy(""),
        initial_service=PromptStrategy("initial service"),
        runner=runner,
        customer_evolver=customer_evolver,
        service_evolver=service_evolver,
        customer_judge=customer_judge,
        service_judge=service_judge,
        domain_policy=runner.service_policy_text,
        output_directory=tmp_path,
        manifest_sha256="manifest",
    )

    assert len(result.generations) == 1
    generation = result.generations[0]
    assert generation["customer_after"]["strategy"] == "Keep negotiating for the requested outcome."
    assert generation["service_after"]["strategy"] == "Verify scope, then complete eligible requests."
    assert generation["service_phase"]["accepted"] is True
    assert generation["service_phase"]["clean_panel"]["catastrophic_regression"] is False
    assert seen_customer_contexts and seen_service_contexts
    customer_phase_calls = [call for call in runner.calls if "customer-" in call["panel_name"]]
    assert customer_phase_calls
    assert all(call["service"].text == "initial service" for call in customer_phase_calls)
    service_challenge_calls = [call for call in runner.calls if call["panel_name"].endswith("service-candidate")]
    assert len(service_challenge_calls) == 1
    assert service_challenge_calls[0]["customer"].text == "Keep negotiating for the requested outcome."
    assert (tmp_path / "generation-0000.json").is_file()
    assert (tmp_path / "checkpoint.json").is_file()


def test_two_generations_continue_from_the_previous_customer_service_pair() -> None:
    tasks = {
        "e": FakeTask("e", "Evolution task", "Customer's original goal."),
        "v": FakeTask("v", "Clean task", "A normal service request."),
    }
    runner = FakeRunner()
    proposal_inputs = []
    service_inputs = []

    def customer_evolver(context, _count):
        proposal_inputs.append(context)
        return [f"Customer strategy {context['generation']}"]

    result = run_alternating_evolution(
        tasks=tasks,
        evolution_task_ids=("e",),
        validation_task_ids=("v",),
        seed=4,
        generations=2,
        customer_candidate_count=1,
        clean_panel_size=1,
        initial_customer=PromptStrategy("customer zero"),
        initial_service=PromptStrategy("service zero"),
        runner=runner,
        customer_evolver=customer_evolver,
        service_evolver=lambda context: (
            service_inputs.append(context)
            or {"analysis": "observed", "strategy": f"service {context['generation'] + 1}"}
        ),
        customer_judge=lambda _context: {"choice": 0, "reason": "challenge"},
        service_judge=lambda _context: {"improved": True, "reason": "fixed"},
        domain_policy=runner.service_policy_text,
    )

    assert len(result.generations) == 2
    assert proposal_inputs[1]["current_service_strategy"] == "service 1"
    assert service_inputs[1]["customer_strategy"] == "Customer strategy 1"
    assert result.customer.text == "Customer strategy 1"
    assert result.service.text == "service 2"


def test_fresh_challenge_uses_only_evidence_given_before_h_is_loaded() -> None:
    tasks = {"e": FakeTask("e", "Evolution task", "Only E scenario."), "v": FakeTask("v", "V", "V scenario.")}
    runner = FakeRunner()
    result = run_alternating_evolution(
        tasks=tasks,
        evolution_task_ids=("e",),
        validation_task_ids=("v",),
        seed=3,
        generations=1,
        customer_candidate_count=1,
        clean_panel_size=1,
        initial_customer=PromptStrategy("base"),
        initial_service=PromptStrategy("service"),
        runner=runner,
        customer_evolver=lambda _context, _count: ["evolved"],
        service_evolver=lambda _context: {"analysis": "none", "strategy": "service"},
        customer_judge=lambda _context: {"choice": 0, "reason": "ok"},
        service_judge=lambda _context: {"improved": False, "reason": "same"},
        domain_policy=runner.service_policy_text,
    )
    contexts = []

    def fresh_evolver(context, count):
        contexts.append(context)
        assert count == 1
        assert all(
            row["task"]["task_id"] != "h"
            for row in context["task_interactions"]
        )
        return ["fresh challenger"]

    fresh = propose_fresh_customer_challenge(
        result,
        fresh_evolver,
        tasks=tasks,
        evolution_task_ids=("e",),
        runner=runner,
        domain_policy=runner.service_policy_text,
        seed=500,
    )
    heldout = {"h": FakeTask("h", "Held-out task", "Never shown during evolution.")}
    h_runner = FakeRunner()
    evaluation = run_final_endpoint_evaluation(
        heldout_tasks=heldout,
        heldout_task_ids=("h",),
        seed=99,
        initial_service=result.initial_service,
        final_service=result.service,
        fresh_customer=fresh,
        runner=h_runner,
    )

    assert fresh.text == "fresh challenger"
    assert len(contexts) == 1
    assert {row["task_id"] for cell in evaluation["cells"] for row in cell["episodes"]} == {"h"}
    assert {call["task_id"] for call in h_runner.calls} == {"h"}
