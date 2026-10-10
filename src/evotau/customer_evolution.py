"""Small Customer-only protocol helpers shared by alternating and E-only diagnosis."""

from .customer_skills import PROTOCOL, CustomerSkill
from .tau_provenance import sha256_json

CUSTOMER_SKILL_PROMPT = """You are EvoTau's task-faithful Customer Evolver, not a jailbreak/red-team user.
Discover open-ended reusable interaction procedures that expose observable Service weaknesses
without changing any original task goal, identity, fact, budget, reservation/order, payment fact,
policy or conditional authorization. Fixed scenario and native guidelines always take precedence.
Explore truthful information disclosure timing, organization of EXISTING multi-step requests,
conditional confirmation and cost explanations, reasonable urgency/persistence, or correction
after actual errors. These are examples, not an enumeration of allowable mechanisms.
Prioritize complex successful E trajectories as potential weakness opportunities; failed controls
and past rejected/legal-but-ineffective attempts inform hypotheses, not established vulnerabilities.
Use task-parameterized instructions, never concrete identifiers, new dates, new amounts, new
requests or changed preferences absent from the scenario. Prompt injection, deception, blanket
consent replacing conditional authorization, prolonged refusal of necessary facts, infinite
questioning, arbitrary early termination, or merely exhausting max_steps are prohibited.
Respond truthfully and promptly to necessary clarification. Stop the procedure when its trigger
ceases, required facts are missing, native goals are resolved, or it conflicts with the scenario.
Intensity controls communication tone only. Genuine pressure must be evaluated by E rollouts.
Return EXACTLY requested_candidates candidates in ONE JSON object {"candidates":[...]}.
Each candidate has exactly: schema_version=1, mechanism (open-ended string), trigger (string),
procedure (array of 2–5 distinct steps), intensity (low|medium|high), stop_conditions (1–5 strings),
hypothesis (string, an unproven prediction), evidence_refs (array).
Each evidence_ref has task_id, seed, trajectory_ref, projected_message_index, message_sha256
copied exactly from a supplied retained E message and its row. Empty refs are allowed for new
exploration; never invent evidence. No V/H or gold actions. All length/token limits in
customer_evolution are binding. Do not repeat a supplied compiled strategy or historical mechanism
with cosmetic wording; explore a materially different lawful interaction procedure.
"""


def generate_candidates(providers, context, count):
    """Parsed contract errors are logged rejections; transport/JSON errors stay fail-closed."""
    from .evolution_candidates import EvolverSchemaError

    if "operator_family_hint" in context:
        from .repair_conditioned_bandit import ARMS
        if context["operator_family_hint"] not in ARMS or context.get("repair_context", {}).get("frontier_status") not in ("repair_pair", "no_repair_pair"):
            raise ValueError("invalid frozen RC operator context")
    try:
        return providers.customers(context, count)
    except EvolverSchemaError as error:
        return {
            "candidates": [],
            "generation_rejection": {
                "candidate_validity": "invalid",
                "pre_rollout_rejection": str(error),
                "diagnostics_ref": getattr(error, "diagnostics_ref", None),
            },
        }


def prepare_candidate(proposal, context, policy):
    row = {
        "customer_protocol": PROTOCOL,
        "candidate_id": sha256_json(proposal)[:20],
        "skill": proposal,
        "candidate_validity": "uncertain",
        "failure_mechanism": "",
        "strategy": {"text": ""},
        "accuracy": None,
        "episodes": [],
        "new_failures": [],
        "recovered_cells": [],
        "invalid_customer_episodes": [],
        "paired_accuracy_delta": None,
        "pre_rollout_rejection": None,
        "eligible_for_selection": False,
        "selected": False,
    }
    try:
        skill = CustomerSkill.from_mapping(
            proposal, evidence_rows=context["task_interactions"], policy=policy
        )
    except (ValueError, TypeError, KeyError) as error:
        row.update(
            candidate_validity="invalid",
            pre_rollout_rejection="structure: " + str(error),
        )
        return row, None
    row.update(
        candidate_id=skill.candidate_id,
        procedure_id=skill.procedure_id,
        failure_mechanism=proposal["mechanism"],
        strategy=skill.compile().to_dict(),
    )
    return row, skill


def evaluate_candidate(
    row,
    skill,
    *,
    context,
    incumbent,
    current_service,
    tasks,
    runner,
    panel,
    stage,
    providers,
    domain_policy,
):
    from .alternating import _accuracy
    from .customer_trajectory_validity import paired_customer_feedback
    from .evolution_candidates import EvolverSchemaError

    if skill is None:
        return row, ()
    semantic_context = {
        "strategy": row["strategy"]["text"],
        "skill": skill.to_dict(),
        "customer_protocol": PROTOCOL,
        "scenarios": [r["task"]["user_scenario"] for r in context["task_interactions"]],
    }

    def validate_static():
        try:
            return providers.validate_customer(semantic_context)
        except EvolverSchemaError as error:
            return {
                "schema_error": str(error),
                "diagnostics_ref": getattr(error, "diagnostics_ref", None),
            }

    verdict = stage("static-validator", semantic_context, validate_static)
    row["semantic_validation"] = verdict
    if "schema_error" in verdict:
        row.update(
            candidate_validity="uncertain",
            pre_rollout_rejection="static validator schema error: "
            + verdict["schema_error"],
        )
        return row, ()
    if not all(
        verdict.get(k) is True
        for k in (
            "preserves_facts",
            "preserves_objective",
            "interaction_only",
            "no_benchmark_leakage",
        )
    ):
        row.update(
            candidate_validity="invalid",
            pre_rollout_rejection="semantic_preservation_failed",
        )
        return row, ()
    runs = panel(skill.compile(), current_service)
    row.update(accuracy=_accuracy(runs), episodes=[r.to_dict() for r in runs])
    validity = review_customer_runs(runs, skill=skill.to_dict(), tasks=tasks,
                                    runner=runner, providers=providers, domain_policy=domain_policy,
                                    policy=context["customer_evolution"], stage=stage)
    row.update(
        paired_customer_feedback(incumbent, runs, validity=validity),
        trajectory_validity=validity,
    )
    row["selection_reason"] = (
        "awaiting strict native-accuracy comparison"
        if validity["status"] == "valid"
        else "whole candidate ineligible: invalid or uncertain E cell"
    )
    return row, runs


def review_customer_runs(runs, *, skill, tasks, runner, providers, domain_policy, policy, stage):
    """Existing complete-cell validity chain, reused at both RC Service endpoints."""
    from .alternating import _context_episodes
    from .customer_skills import proxy_tokens
    from .customer_trajectory_validity import (
        TRAJECTORY_VALIDATOR_PROMPT,
        assess_trajectories,
        trajectory_context,
    )
    from .evolution_candidates import EvolverSchemaError
    review_context = trajectory_context(
        _context_episodes(runs, runner, tasks), skill, domain_policy
    )
    import json

    cells = []
    # One complete cell per request: no silent truncation, individually resumable.
    for cell in review_context["cells"]:
        subcontext = {**review_context, "cells": [cell]}

        def review(subcontext=subcontext, cell=cell):
            count = proxy_tokens(
                TRAJECTORY_VALIDATOR_PROMPT + json.dumps(subcontext, ensure_ascii=False)
            )
            if count > policy["max_context_tokens"]:
                return {
                    "status": "uncertain",
                    "cells": [
                        {
                            "task_id": cell["task_id"],
                            "seed": cell["seed"],
                            "status": "uncertain",
                            "reason": "complete trajectory exceeds frozen review allowance",
                            "evidence_message_indices": [],
                        }
                    ],
                }
            try:
                return assess_trajectories(
                    subcontext, providers.validate_customer_trajectory
                )
            except EvolverSchemaError as error:
                return {
                    "status": "uncertain",
                    "cells": [
                        {
                            "task_id": cell["task_id"],
                            "seed": cell["seed"],
                            "status": "uncertain",
                            "reason": "trajectory schema error: " + str(error),
                            "evidence_message_indices": [],
                        }
                    ],
                    "diagnostics_ref": getattr(error, "diagnostics_ref", None),
                }

        decision = stage(
            "trajectory-validator-" + sha256_json([cell["task_id"], cell["seed"]])[:12],
            subcontext,
            review,
        )
        cells.extend(decision["cells"])
    validity = {
        "status": "invalid"
        if any(c["status"] == "invalid" for c in cells)
        else "uncertain"
        if any(c["status"] == "uncertain" for c in cells)
        else "valid",
        "cells": cells,
    }
    return validity


def propose_fresh_customer_skill(
    result,
    providers,
    *,
    tasks,
    runner,
    domain_policy,
    output_directory,
    manifest_sha256,
    policy,
    max_parallel_episodes,
):
    """E-only static and observed validity before H is unsealed; never an archive fallback."""
    from pathlib import Path

    from .alternating import _context_episodes, _run_panel, _write_json_once
    from .evolution_artifacts import EvolutionJournal
    from .evolution_context import build_customer_evidence
    from .records import EpisodeRecord

    providers.configure_customer({"customer_evolution": policy})
    journal = EvolutionJournal(output_directory, manifest_sha256)
    ids = tuple(r.task_id for r in result.final_evolution_episodes)
    if len(set(ids)) != len(ids):
        raise ValueError("fresh Customer requires one frozen fitness seed per E task")
    e_tasks = {k: tasks[k] for k in ids}
    context = {
        "task_interactions": build_customer_evidence(
            _context_episodes(result.final_evolution_episodes, runner, e_tasks),
            representative_cases=policy["representative_cases"],
            case_chars=policy["case_chars"],
            customer_protocol=PROTOCOL,
        ),
        "customer_evolution": policy,
        "customer_attempt_history": result.generations[-1].get(
            "customer_attempt_history", []
        ),
        "current_customer_strategy": result.customer.text,
        "service_policy": domain_policy,
        "purpose": "fresh E-only proposal before H loads",
    }
    proposals = journal.freeze(
        "final-fresh-customer-skill",
        context,
        lambda: generate_candidates(providers, context, 1),
    )
    row, skill = (
        prepare_candidate(proposals["candidates"][0], context, policy)
        if proposals["candidates"]
        else (
            {
                **proposals["generation_rejection"],
                "strategy": {"text": ""},
                "eligible_for_selection": False,
            },
            None,
        )
    )

    def panel(c, s):
        inputs = {
            "customer": c.to_dict(),
            "service": s.to_dict(),
            "task_ids": ids,
            "seed": result.final_evolution_episodes[0].seed,
        }
        saved = journal.freeze(
            "final-fresh-customer-E-rollout",
            inputs,
            lambda: {
                "episodes": [
                    r.to_dict()
                    for r in _run_panel(
                        runner,
                        tasks=e_tasks,
                        task_ids=ids,
                        seed=inputs["seed"],
                        customer=c,
                        service=s,
                        panel_name="fresh-customer-validity-E",
                        max_parallel_episodes=max_parallel_episodes,
                    )
                ]
            },
        )
        return tuple(EpisodeRecord.from_dict(r) for r in saved["episodes"])

    row, _ = evaluate_candidate(
        row,
        skill,
        context=context,
        incumbent=result.final_evolution_episodes,
        current_service=result.service,
        tasks=e_tasks,
        runner=runner,
        panel=panel,
        stage=lambda name, inputs, cb: journal.freeze(
            "final-fresh-customer-" + name, inputs, cb
        ),
        providers=providers,
        domain_policy=domain_policy,
    )
    prior = {
        result.customer.text,
        result.initial_customer.text,
        *(e["strategy"] for e in result.generations[-1].get("customer_archive", [])),
    }
    distinct = row["strategy"]["text"] not in prior
    customer = skill.compile() if row["eligible_for_selection"] and distinct else None
    _write_json_once(
        Path(output_directory) / "fresh-customer-proposal.json",
        {
            **row,
            "schema_version": 4,
            "manifest_sha256": manifest_sha256,
            "status": "available" if customer else "unavailable",
            "selection_protocol": "one structured E-only proposal; static + full E trajectory validity; distinct; no H feedback",
            "rejection_reason": None
            if customer
            else row.get("pre_rollout_rejection")
            or ("not_distinct" if not distinct else "trajectory_validity_failed"),
        },
    )
    return customer
