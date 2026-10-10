"""V2 orchestration: direct mutation/diversify/screen/gate, isolated from legacy V1."""

import json
import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from .evolution_archive import (
    CustomerChallengeArchive,
    EvolutionArchive,
    crossover_eligible,
    semantic_duplicate,
)
from .evolution_artifacts import EvolutionJournal
from .evolution_candidates import validate_direct_evidence, validate_mutation
from .evolution_failures import episode_metrics, paired_effect, paired_failure_matrix
from .evolution_gate import cheap_screen, evaluate_gate, pass_power_k
from .evolution_history import stagnation_state, summarize_effects
from .records import EpisodeRecord, customer_strategy_id, service_strategy_id
from .service_skills import (
    ServiceSkillMemoryV2,
    apply_v2_mutation,
    service_skill_id_high_watermark,
    validate_skill_budgets,
)
from .strategies import PromptStrategy
from .tau_provenance import sha256_json


def _candidate_outcome(callback, context):
    """Reject parsed candidate contract errors; infrastructure errors still propagate."""
    from .evolution_candidates import EvolverSchemaError
    from .evolver_recovery import RecoveryExhausted
    from .provider_diagnostics import safe_error

    try:
        mutation = callback()
    except RecoveryExhausted as error:
        return {"status": "REJECTED", "error_type": "RECOVERY_EXHAUSTED",
                "reason": str(error), "recovery": error.outcome,
                "diagnostics_ref": error.diagnostics_ref}
    except EvolverSchemaError as error:
        return {
            "status": "REJECTED",
            "error_type": "EvolverSchemaError",
            "reason": safe_error(error),
            "diagnostics_ref": getattr(error, "diagnostics_ref", None),
        }
    # Validate only the returned candidate here, never catch dispatch/runtime ValueErrors.
    try:
        validate_mutation(mutation)
        validate_direct_evidence(mutation, context)
        from .failure_analysis import validate_assigned_mutation
        validate_assigned_mutation(mutation, context)
    except (ValueError, TypeError, KeyError) as error:
        return {
            "status": "REJECTED",
            "error_type": "CandidateContractError",
            "reason": safe_error(error),
            "raw_candidate": mutation,
        }
    return {"status": "VALID", "mutation": mutation}


def _rejected_candidate(ident, outcome, generation, parents=()):
    return {
        "mutation_id": ident,
        "generation": generation,
        "mutation": None,
        "hypothesis": None,
        "parent_mutation_ids": list(parents),
        "runtime_deployed": False,
        "screen": None,
        "gate": None,
        "effect": None,
        "decision": "REJECTED",
        "pre_rollout_rejection": "candidate_contract_error",
        "candidate_error": outcome,
    }


def _document(role, strategy):
    return {
        "role": role,
        "carrier": "skill_memory_v2" if role == "service" else "prompt_strategy",
        "strategy_id": service_strategy_id(strategy)
        if role == "service"
        else customer_strategy_id(strategy),
        "strategy": strategy.to_dict() if role == "service" else strategy.text,
        **({"skill_count": len(strategy.skills)} if role == "service" else {}),
    }


def _evolution_parent(entry):
    """E-derived mutation evidence only; never expose raw V gate cells to Evolvers."""
    return {
        key: deepcopy(entry[key])
        for key in (
            "mutation_id",
            "mutation",
            "effect",
            "generation",
            "evaluation_scope",
        )
        if key in entry
    }


def run_skill_evolution_v2(
    *,
    tasks,
    evolution_task_ids,
    validation_task_ids,
    seed,
    generations,
    customer_candidate_count,
    max_parallel_episodes,
    evolution_fitness_seed,
    run_validation,
    initial_customer,
    initial_service,
    runner,
    providers,
    domain_policy,
    output_directory,
    checkpoint_path,
    manifest_sha256,
    policy,
    request_budget=None,
    episode_job_telemetry=None,
    interrupt_after_stage=None,
):
    from .alternating import (
        AlternatingResult,
        _accuracy,
        _context_episodes,
        _run_panel,
        _service_context_episodes,
        _write_json_atomic,
        _write_json_once,
    )

    if policy.get("algorithm_version") not in ("direct_skill_evolution_v1", "direct_skill_v_validation_v2", "analyst_skill_v_validation_v3", "analyst_skill_recovery_v4"):
        raise ValueError(
            "Legacy Diagnoser configuration is read-only; create a versioned Direct Skill config"
        )
    from .skill_evolution_config import (
        evaluation_seed_schedule,
        validate_v2_promotion_readiness,
    )
    validate_v2_promotion_readiness(policy, run_validation=run_validation)
    analyst_mode = policy["algorithm_version"] in ("analyst_skill_v_validation_v3", "analyst_skill_recovery_v4")
    v_primary = policy["evaluation"].get("promotion_protocol") == "v_primary"
    if v_primary and (not run_validation or policy["statistical_gate"]["method"] != "task_block_bootstrap"
                      or not policy["statistical_gate"]["enabled"]):
        raise ValueError("V-primary promotion requires enabled V task-block statistical gates")
    def prompt_parent(entry):
        value = _evolution_parent(entry)
        if (v_primary or policy.get("targeted_repair")) and value.get("effect"):
            value["effect"]["rejection_reason"] = "deployment outcome withheld; E effects only"
        return value

    root = Path(output_directory)
    if hasattr(providers, "configure_recovery"):
        providers.configure_recovery(policy, root, manifest_sha256)
    journal = EvolutionJournal(root, manifest_sha256)
    e, v = tuple(map(str, evolution_task_ids)), tuple(map(str, validation_task_ids))
    if set(tasks) - (set(e) | (set(v) if run_validation else set())) or not set(
        e
    ) <= set(tasks):
        raise ValueError("V2 task universe must be exactly unsealed E/V")
    if not isinstance(initial_service, ServiceSkillMemoryV2):
        raise TypeError("V2 requires V2 runtime memory")
    if run_validation and (not v or not set(v) <= set(tasks) or set(e) & set(v)):
        raise ValueError("formal V2 requires a disjoint loaded V panel")
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.is_symlink():
        raise ValueError("unsafe V2 checkpoint")
    saved = (
        json.loads(checkpoint_file.read_text()) if checkpoint_file.exists() else None
    )
    if saved is not None and (
        saved.get("manifest_sha256") != manifest_sha256
        or saved.get("checkpoint_sha256")
        != sha256_json({k: w for k, w in saved.items() if k != "checkpoint_sha256"})
    ):
        raise ValueError("V2 checkpoint manifest/digest mismatch")
    customer, service = initial_customer, initial_service
    effects, matrix, documents, provenance = [], [], [], []
    archive = EvolutionArchive(max_candidates=policy["archive"]["max_candidates"])
    customers = CustomerChallengeArchive()
    from .customer_skills import PROTOCOL
    from .evolution_archive import CustomerAttemptHistory
    customer_policy = policy.get("customer_evolution")
    customer_attempts = CustomerAttemptHistory(max_history=(customer_policy or {}).get("max_history", 20))
    if saved is not None and saved["state"].get("customer_protocol") != (PROTOCOL if customer_policy else None):
        raise ValueError("checkpoint Customer protocol mismatch")
    rc_policy = policy.get("repair_conditioned_bandit") or policy.get("open_repair_search")
    rc, repair_pair = None, None
    if rc_policy:
        from .customer_diagnostic import RepairConditionedTrials
        from .repair_conditioned_bandit import PROTOCOL as RC_PROTOCOL
        if not customer_policy:
            raise ValueError("RC-Bandit requires Customer Skill v1")
        if customer_candidate_count > rc_policy["max_trials_per_generation"]:
            raise ValueError("Customer candidate count exceeds frozen RC trial budget")
        if policy.get("open_repair_search"):
            from .open_repair_search import PROTOCOL as RC_PROTOCOL
            from .open_repair_search import OpenRepairTrials as RepairConditionedTrials
        rc = RepairConditionedTrials(policy=rc_policy, customer_policy=customer_policy,
            tasks={k: tasks[k] for k in e}, runner=runner, providers=providers,
            domain_policy=domain_policy, root=root, manifest_sha=manifest_sha256, budget=request_budget)
    discovery_archive = None
    targeted_policy = policy.get("targeted_repair")
    if policy.get("discovery_handoff"):
        if not rc:
            raise ValueError("discovery handoff requires repair search")
        from .repair_discovery_archive import RepairDiscoveryArchive
        discovery_archive = RepairDiscoveryArchive(root, manifest_sha256, e)
        rc.discovery_handoff = True
    if targeted_policy and not run_validation:
        raise ValueError("targeted promotion requires independent V; E-only discovery runs cannot deploy")
    if saved is not None and saved["state"].get("bandit_protocol") != (RC_PROTOCOL if rc else None):
        raise ValueError("checkpoint Bandit protocol mismatch")

    def rc_guard(callback):
        try:
            return callback()
        except Exception as error:
            if rc:
                rc.failure(error)
            raise

    watermark = service_skill_id_high_watermark(service)
    final_runs = ()
    fitness_seed = seed if evolution_fitness_seed is None else evolution_fitness_seed
    evo = policy["service_evolution"]
    smoke = not run_validation
    timing_path = root / "evolution-timing.json"
    timing = (
        json.loads(timing_path.read_text())
        if timing_path.exists()
        else {"manifest_sha256": manifest_sha256, "stages": {}}
    )
    if timing.get("manifest_sha256") != manifest_sha256:
        raise ValueError("timing manifest mismatch")

    def stage(g, name, inputs, callback):
        _write_json_atomic(
            root / f"generation-{g:04d}-stage.json",
            {
                "schema_version": 3,
                "manifest_sha256": manifest_sha256,
                "generation": g,
                "stage": name,
                "panel_task_ids": inputs.get("task_ids")
                if isinstance(inputs, dict)
                else None,
                "seed_schedule": inputs.get("seeds")
                if isinstance(inputs, dict)
                else None,
            },
        )

        def measured_callback():
            started = perf_counter()
            try:
                if hasattr(providers, "last_recovery"):
                    providers.last_recovery = None
                value = callback()
                recovery = getattr(providers, "last_recovery", None)
                if recovery is not None and isinstance(value, dict):
                    value = {**value, "recovery": recovery}
                    if value.get("status") == "VALID" and recovery["status"] == "RECOVERED":
                        value["status"] = "RECOVERED"
                return value
            finally:
                key = f"g{g:04d}-{name}"
                timing["stages"][key] = (
                    timing["stages"].get(key, 0) + perf_counter() - started
                )
                _write_json_atomic(timing_path, timing)

        output = journal.freeze(f"g{g:04d}-{name}", inputs, measured_callback)
        if interrupt_after_stage == name:
            raise RuntimeError(f"deterministic interruption after {name}")
        return output

    def panel(g, name, ids, seeds, c, s):
        inputs = {
            "task_ids": list(ids),
            "seeds": list(seeds),
            "customer": None if c is None else c.to_dict(),
            "service": s.to_dict(),
        }

        def run():
            records = []
            for fitness in seeds:
                records.extend(
                    _run_panel(
                        runner,
                        task_ids=ids,
                        tasks=tasks,
                        seed=fitness,
                        customer=c,
                        service=s,
                        panel_name=f"generation-{g}-{name}-seed-{fitness}",
                        max_parallel_episodes=max_parallel_episodes,
                        telemetry=episode_job_telemetry,
                    )
                )
            _accuracy(records)  # No incomplete or unknown outcome becomes fitness.
            return {"episodes": [r.to_dict() for r in records]}

        return tuple(
            EpisodeRecord.from_dict(r) for r in stage(g, name, inputs, run)["episodes"]
        )

    def service_evidence(records):
        result = _service_context_episodes(records, runner, tasks)
        for row in result:
            row["task"] = {
                "task_id": row["task"]["task_id"]
            }  # descriptions can contain hidden intent
        return result

    for g in range(generations):
        before_c, before_s = customer, service
        frozen_start = {
            "customer": customer.to_dict(),
            "service": service.to_dict(),
            "effects_sha256": sha256_json(effects),
            "policy": policy,
            "generation": g,
        }
        commit_path = journal.root / f"g{g:04d}-generation_complete.json"
        if commit_path.exists():
            commit = journal.freeze(
                f"g{g:04d}-generation_complete", frozen_start, lambda: None
            )
            state = commit["state"]
            customer, service = (
                PromptStrategy(state["customer"]["text"]),
                ServiceSkillMemoryV2.from_mapping(state["service"]),
            )
            effects, matrix, provenance, watermark = (
                state["mutation_effects"],
                state["failure_matrix"],
                state["service_provenance"],
                state["skill_id_high_watermark"],
            )
            archive = EvolutionArchive(
                state["evolution_archive"]["entries"],
                max_candidates=policy["archive"]["max_candidates"],
            )
            customers = CustomerChallengeArchive(state["customer_archive"])
            if customer_policy:
                if state.get("customer_protocol") != PROTOCOL:
                    raise ValueError("generation Customer protocol mismatch")
                customer_attempts = CustomerAttemptHistory(state["customer_attempt_history"], max_history=customer_policy["max_history"])
            if rc:
                if state.get("bandit_protocol") != RC_PROTOCOL:
                    raise ValueError("generation Bandit protocol mismatch")
                rc.state = rc.restore_state(state["bandit_state"])
                repair_pair = state["repair_service_pair"]
                if policy.get("open_repair_search"):
                    rc.repair_evidence = state["open_repair_evidence"]
            documents.append(commit["generation"])
            _write_json_once(root / f"generation-{g:04d}.json", commit["generation"])
            final_runs = tuple(
                EpisodeRecord.from_dict(r) for r in state["final_evolution_episodes"]
            )
            if (
                saved is not None
                and saved["completed_generation"] == g
                and saved["state"] != state
            ):
                raise ValueError("checkpoint differs from committed generation")
            continue
        if saved is not None and saved["completed_generation"] >= g:
            raise ValueError("checkpoint refers to missing generation commit")
        incumbent = panel(g, "customer_incumbent", e, [fitness_seed], customer, service)
        history = (
            summarize_effects(effects, max_families=policy["history"]["max_families"])
            if policy["history"]["summarize"]
            else [prompt_parent(entry) for entry in effects]
        )
        from .evolution_context import (
            CUSTOMER_EVIDENCE_INSTRUCTIONS,
            build_customer_evidence,
        )

        c_context = {
            "generation": g,
            "task_interactions": build_customer_evidence(
                _context_episodes(incumbent, runner, tasks),
                representative_cases=(customer_policy or policy["mutation_context"])["representative_cases"],
                case_chars=(customer_policy or policy["mutation_context"])["case_chars"],
                **({"customer_protocol": PROTOCOL} if customer_policy else {}),
            ),
            "evidence_instructions": CUSTOMER_EVIDENCE_INSTRUCTIONS,
            "current_customer_strategy": customer.text,
            "current_service": service.to_dict(),
            "incumbent_accuracy": _accuracy(incumbent),
            "service_policy": domain_policy,
            "challenge_archive": customers.entries,
        }
        if customer_policy:
            c_context.update(customer_evolution=customer_policy, customer_attempt_history=customer_attempts.entries)
            # Full replay strategies remain archived, not duplicated unboundedly in prompts.
            c_context["challenge_archive"] = [{k: entry.get(k) for k in
                ("strategy_id", "semantic_family", "accuracy", "accepted", "generation")}
                for entry in customers.entries[-customer_policy["max_history"]:]]
            from .customer_evolution import generate_candidates
        if rc:
            if repair_pair and repair_pair["after_id"] != service_strategy_id(before_s):
                raise ValueError("repair frontier does not match current deployed Service")
            # One immutable pull per independent proposal, with sequential updates.
            generated = {"candidates": []}
            def rc_proposals(g=g, c_context=c_context, repair_pair=repair_pair):
                for index in range(customer_candidate_count):
                    context = rc_guard(lambda index=index, g=g, c_context=c_context, repair_pair=repair_pair: rc.begin(g, index, c_context, repair_pair,
                        lambda name, inputs, cb, g=g: stage(g, name, inputs, cb)))
                    proposal_result = rc_guard(lambda index=index, context=context, g=g: stage(g,
                        f"rc-{index}-proposal", context,
                        lambda context=context: generate_candidates(providers, context, 1)))
                    if proposal_result.get("generation_rejection"):
                        yield {"rc_generation_rejection": proposal_result["generation_rejection"]}
                    elif len(proposal_result["candidates"]) != 1:
                        rc_guard(lambda: (_ for _ in ()).throw(ValueError("RC pull requires exactly one candidate")))
                    else:
                        yield proposal_result["candidates"][0]
            proposals = rc_proposals()
        else:
            generated = stage(
                g,
                "customer_candidates",
                c_context,
                lambda c_context=c_context: (generate_candidates(providers, c_context, customer_candidate_count)
                                            if customer_policy else providers.customers(c_context, customer_candidate_count)),
            )
            proposals = generated["candidates"]
        candidate_rows, selected, selected_accuracy = (
            [],
            incumbent,
            _accuracy(incumbent),
        )
        selected_source = "incumbent"
        if customer_policy:
            from .customer_evolution import (
                evaluate_candidate as evaluate_customer_candidate,
            )
            from .customer_evolution import prepare_candidate
            if generated.get("generation_rejection"):
                failure = generated["generation_rejection"]
                candidate_rows.append(failure)
            tested_texts = {entry["strategy"] for entry in customers.entries}
            tested_procedures = {entry.get("procedure_id") for entry in customer_attempts.entries}
            for i, proposal in enumerate(proposals):
                if "rc_generation_rejection" in proposal and rc:
                    row = {**proposal["rc_generation_rejection"], "eligible_for_selection": False,
                           "strategy": {"text": ""}, "accuracy": None, "episodes": []}
                    skill = None
                else:
                    row, skill = prepare_candidate(proposal, c_context, customer_policy)
                if skill is not None and (skill.compile().text == customer.text or
                        skill.compile().text in tested_texts or skill.procedure_id in tested_procedures):
                    row.update(candidate_validity="invalid", pre_rollout_rejection="duplicate compiled Customer strategy")
                    skill = None
                row, runs = rc_guard(lambda row=row, skill=skill, i=i, g=g, c_context=c_context, incumbent=incumbent, before_s=before_s: evaluate_customer_candidate(
                    row, skill, context=c_context, incumbent=incumbent, current_service=before_s,
                    tasks={k: tasks[k] for k in e}, runner=runner, providers=providers,
                    domain_policy=domain_policy,
                    panel=lambda c, svc, i=i, g=g: panel(g, f"customer-candidate-{i}", e, [fitness_seed], c, svc),
                    stage=lambda name, inputs, cb, i=i, g=g: stage(g, f"customer-{i}-{name}", inputs, cb),
                ))
                if rc:
                    row["bandit_trial"] = rc_guard(lambda row=row, skill=skill, repair_pair=repair_pair, i=i, g=g: rc.finish(row, skill, pair=repair_pair,
                        stage=lambda name, inputs, cb, i=i, g=g: stage(g, f"rc-{i}-{name}", inputs, cb),
                        panel=lambda name, seeds, c, svc, i=i, g=g: panel(g, f"rc-{i}-{name}-E", e, seeds, c, svc)))
                if row["eligible_for_selection"]:
                    accuracy = row["accuracy"]
                    if accuracy < selected_accuracy:
                        customer, selected, selected_accuracy, selected_source = skill.compile(), runs, accuracy, i
                    customers.add({"strategy": row["strategy"]["text"], "semantic_family": row["failure_mechanism"],
                                   "target_weakness_family": skill.to_dict()["hypothesis"],
                                   "substantive_delta_from_prior": "structured task-faithful procedure"}, accuracy, g, False, True)
                    tested_texts.add(row["strategy"]["text"])
                candidate_rows.append(row)
                if skill is not None:
                    tested_procedures.add(skill.procedure_id)
            for row in candidate_rows:
                row["selected"] = bool(row.get("eligible_for_selection") and row.get("strategy", {}).get("text") == customer.text)
                if row.get("eligible_for_selection"):
                    row["selection_reason"] = ("selected: strict native E accuracy decrease" if row["selected"] else
                        "no strict native E accuracy decrease" if row["accuracy"] >= _accuracy(incumbent) else
                        "another eligible candidate has lower native E accuracy")
                customer_attempts.add(row, g)
        else:
            for i, proposal in enumerate(proposals):
                context = {
                    "strategy": proposal["strategy"],
                    "scenarios": [
                        row["task"]["user_scenario"]
                        for row in c_context["task_interactions"]
                    ],
                }
                verdict = stage(
                    g,
                    f"customer-validator-{i}",
                    context,
                    lambda ctx=context: providers.validate_customer(ctx),
                )
                flags = (
                    "preserves_facts",
                    "preserves_objective",
                    "interaction_only",
                    "no_benchmark_leakage",
                )
                valid = all(verdict.get(k) is True for k in flags)
                row = {
                    **proposal,
                    "strategy": {"text": proposal["strategy"]},
                    "semantic_validation": verdict,
                    "accuracy": None,
                    "episodes": [],
                    "pre_rollout_rejection": None
                    if valid
                    else "semantic_preservation_failed",
                }
                if valid:
                    challenge = PromptStrategy(proposal["strategy"])
                    runs = panel(
                        g, f"customer-candidate-{i}", e, [fitness_seed], challenge, before_s
                    )
                    accuracy = _accuracy(runs)
                    row.update(accuracy=accuracy, episodes=[r.to_dict() for r in runs])
                    if accuracy < selected_accuracy:
                        customer, selected, selected_accuracy, selected_source = (
                            challenge,
                            runs,
                            accuracy,
                            i,
                        )
                    customers.add(proposal, accuracy, g, False, True)
                candidate_rows.append(row)
        generation_discoveries = []
        if discovery_archive:
            for row in candidate_rows:
                trial = row.get("bandit_trial")
                if trial and trial["reward"]:
                    generation_discoveries.extend(discovery_archive.publish(
                        trial, row["skill"], repair_pair, trial["review_context"],
                        selected=row["selected"], trial_ref=rc.ledger.root / f"{trial['trial_id']}-final.json",
                        customer_policy=customer_policy, evidence_rows=c_context["task_interactions"],
                        min_replications=rc.policy["min_replications"]))
        generation_discoveries.sort(key=lambda d: (d["trial_id"], d["discovery_id"]))
        frozen_targets = generation_discoveries[:targeted_policy["max_targets_per_generation"]] if targeted_policy else []
        if discovery_archive:
            stage(g, "discovery_handoff_selection", {"discovery_ids": [d["discovery_id"] for d in generation_discoveries],
                  "targeted_policy": targeted_policy},
                  lambda frozen_targets=frozen_targets: {"frozen_target_ids": [d["discovery_id"] for d in frozen_targets]})
        for entry in customers.entries:
            entry["accepted"] = entry["strategy"] == customer.text
        stage(
            g,
            "customer_selection",
            {"rows": candidate_rows, "before": before_c.to_dict()},
            lambda selected_source=selected_source, selected_accuracy=selected_accuracy, customer=customer: {
                "selected_source": selected_source,
                "selected_accuracy": selected_accuracy,
                "customer": customer.to_dict(),
            },
        )
        # All E cells are visible; only bounded original evidence is expanded.
        from .evolution_context import (
            SERVICE_MUTATION_EVIDENCE_INSTRUCTIONS,
            build_service_mutation_evidence,
            enforce_mutation_context_budget,
        )

        all_evidence = build_service_mutation_evidence(
            service_evidence(selected),
            representative_cases=policy["mutation_context"]["representative_cases"],
            case_chars=policy["mutation_context"]["case_chars"],
        )
        if v_primary or targeted_policy:
            for family in history:
                if family.get("effect"):
                    family["effect"]["rejection_reason"] = "deployment outcome withheld; E effects only"
                for key in ("successful_mechanisms", "rejected_mechanisms"):
                    for mechanism in family.get(key, []):
                        mechanism["reason"] = "deployment outcome withheld; E effects only"
        s_context = {
            "algorithm_version": policy["algorithm_version"],
            "evidence_budget": policy["mutation_context"],
            "generation": g,
            "task_interactions": all_evidence,
            "evidence_instructions": SERVICE_MUTATION_EVIDENCE_INSTRUCTIONS,
            "failure_matrix": [
                {k: row[k] for k in ("task_id", "seed", "candidate_id", "status", "old_success", "new_success", "termination_reason", "activated_skill_ids")}
                if v_primary or targeted_policy else row
                for row in matrix if row["task_id"] in e
            ][-len(e) * 8 :],
            "current_service_memory": before_s.to_dict(),
            "service_policy": domain_policy,
            "interaction_metrics": episode_metrics(selected),
            "current_outcomes": [
                {
                    key: r.to_dict()[key]
                    for key in (
                        "task_id",
                        "seed",
                        "status",
                        "task_success",
                        "native_reward",
                        "termination_reason",
                        "total_steps",
                        "tool_calls",
                    )
                }
                for r in selected
            ],
            "previous_mutation_effects": history if evo["lineage"] else [],
            "prior_fixed_cases": sorted(
                {t for ent in effects for t in ent["effect"]["fail_to_pass"] if t in e}
            ),
            "prior_broken_cases": sorted(
                {t for ent in effects for t in ent["effect"]["pass_to_fail"] if t in e}
            ),
            "exploration": stagnation_state(
                documents, effects, evo["stagnation_patience"]
            ),
        }
        if generation_discoveries:
            from .repair_discovery_archive import service_handoff
            handoff = service_handoff(generation_discoveries, rc.state["trials"], runner,
                                      {k: tasks[k] for k in e}, policy["mutation_context"])
            s_context["discovery_repair_targets"] = handoff
            s_context["frozen_target_ids"] = [d["discovery_id"] for d in frozen_targets]
            for d in generation_discoveries:
                discovery_archive.event(d["discovery_id"], g, "handoff",
                    {"used_for_service_prompt": True, "frozen_evaluation_target": d in frozen_targets,
                     "training_panel": "E", "projection_sha256": sha256_json(handoff),
                     "source_ref": f"generation-{g:04d}.json", "repair_outcome": "not_yet_evaluated"})
            # Original validator registry includes every supplied target trajectory.
            s_context["task_interactions"] = all_evidence + [r for d in handoff for r in d["task_interactions"]]
        board = []
        looks = evo["candidates_per_generation"] + int(evo["crossover"])
        replay = policy["opponent_replay"]
        opponents = [("current", customer, replay["current_weight"])]
        if replay["enabled"]:
            opponents.extend(
                (entry["strategy_id"], PromptStrategy(entry["strategy"]), 1.0)
                for entry in customers.replay(
                    replay["archived_customers"], g, customer.text
                )
            )
            if replay["include_native_customer"]:
                opponents.append(("native", None, 1.0))
        looks *= len(opponents) + int(not smoke and not v_primary)
        if v_primary:
            # Frozen upper bound includes every candidate, generation and possible opponent.
            looks = generations * (evo["candidates_per_generation"] + int(evo["crossover"])) * (
                1 + (replay["archived_customers"] + int(replay["include_native_customer"]) if replay["enabled"] else 0))
        all_proposals = []
        rejected_proposals = []

        def candidate_hypothesis(mutation):
            return {
                "cluster_id": mutation["target_cluster_id"],
                "root_cause": mutation["root_cause_hypothesis"],
                "root_cause_sha256": sha256_json(mutation["root_cause_hypothesis"]),
                "source": "assigned_analyst_hypothesis" if analyst_mode else "direct_candidate",
                "evidence_task_ids": mutation["evidence_task_ids"],
                "protected_success_task_ids": mutation["protected_success_task_ids"],
            }

        analyst_outcome = None
        hypothesis_selection = None
        if analyst_mode:
            from .evolution_candidates import FAILURE_ANALYST_PROMPT
            from .failure_analysis import (
                analysis_outcome,
                conservative_dedup_fallback,
                select_distinct_hypotheses,
            )
            analyst_context = {**s_context, "requested_hypotheses": evo["candidates_per_generation"]}
            enforce_mutation_context_budget(
                analyst_context, FAILURE_ANALYST_PROMPT, policy["mutation_context"]["max_proxy_tokens"]
            )
            analyst_outcome = stage(
                g, "service_failure_analysis", analyst_context,
                lambda ctx=analyst_context: analysis_outcome(lambda: providers.analyze_service_failures(ctx), ctx),
            )
            hypotheses = analyst_outcome["hypotheses"]
            review_context = {"algorithm_version": policy["algorithm_version"], "hypotheses": hypotheses}
            if len(hypotheses) > 1:
                from .evolution_candidates import MECHANISM_DEDUP_PROMPT
                from .failure_analysis import diversity_outcome
                enforce_mutation_context_budget(review_context, MECHANISM_DEDUP_PROMPT, policy["mutation_context"]["max_proxy_tokens"])
                review_outcome = stage(
                    g, "service_mechanism_diversity", review_context,
                    lambda ctx=review_context, hs=hypotheses: diversity_outcome(lambda: providers.deduplicate_failure_hypotheses(ctx), hs),
                )
            else:
                review_outcome = {"status": "VALID", "comparisons": []}
            hypothesis_selection = (
                select_distinct_hypotheses(hypotheses, {"comparisons": review_outcome["comparisons"]}, evo["candidates_per_generation"])
                if review_outcome["status"] in ("VALID", "RECOVERED")
                else conservative_dedup_fallback(hypotheses, review_outcome)
                if review_outcome["status"] == "DEGRADED"
                else {"assigned_hypotheses": [], "excluded_hypotheses": [], "semantic_distinctness_proven": False}
            )
            hypothesis_selection["diversity_review"] = review_outcome
            hypothesis_selection = stage(
                g, "service_hypothesis_assignment",
                {"analysis": analyst_outcome, "review": review_outcome, "candidate_budget": evo["candidates_per_generation"]},
                lambda selection=hypothesis_selection: selection,
            )
            assignments = hypothesis_selection["assigned_hypotheses"]
        else:
            assignments = [None] * evo["candidates_per_generation"]

        for bias, assigned in enumerate(assignments):
            ctx = {**s_context, "proposal_index": bias}
            if analyst_mode:
                ctx["assigned_hypothesis"] = assigned
                from .evolution_candidates import MUTATOR_PROMPT as proposal_prompt
            else:
                ctx.update({
                    "proposal_bias": ("narrow_applicability", "minimal_behavior", "structural_decomposition")[bias % 3],
                    "archive_parent": (
                        prompt_parent(archive.entries[bias % len(archive.entries)])
                        if archive.entries and s_context["exploration"]["explore"] else None
                    ),
                })
                from .evolution_candidates import (
                    DIRECT_MUTATOR_PROMPT as proposal_prompt,
                )
            enforce_mutation_context_budget(ctx, proposal_prompt, policy["mutation_context"]["max_proxy_tokens"])
            ident = f"g{g:04d}-direct-{bias}"
            if evo.get("candidate_error_policy", "fail_closed") == "reject_candidate":
                outcome = stage(
                    g,
                    f"service_proposals-{ident}",
                    ctx,
                    lambda ctx=ctx: _candidate_outcome(
                        lambda: providers.propose_skill_mutation(ctx), ctx
                    ),
                )
                if outcome["status"] == "REJECTED":
                    rejected_proposals.append(
                        (_rejected_candidate(ident, outcome, g), ())
                    )
                    continue
                proposal = outcome["mutation"]
            else:
                proposal = stage(
                    g,
                    f"service_proposals-{ident}",
                    ctx,
                    lambda ctx=ctx: providers.propose_skill_mutation(ctx),
                )
                validate_mutation(proposal)
                validate_direct_evidence(proposal, ctx)
                from .failure_analysis import validate_assigned_mutation
                validate_assigned_mutation(proposal, ctx)
            all_proposals.append((ident, proposal, candidate_hypothesis(proposal), ()))

        seen_candidates = []

        def evaluate_candidate(
            ident,
            mutation,
            cluster,
            parents,
            *,
            g=g,
            before_s=before_s,
            effects=effects,
            customer=customer,
            opponents=opponents,
            looks=looks,
            matrix=matrix,
            seen_candidates=seen_candidates,
            frozen_targets=frozen_targets,
        ):
            nonlocal watermark
            target, protected = (
                set(cluster["evidence_task_ids"]),
                set(cluster["protected_success_task_ids"]),
            )
            row = {
                "mutation_id": ident,
                "mutation": mutation,
                "hypothesis": cluster,
                "generation": g,
                "parent_mutation_ids": list(parents),
                "runtime_deployed": False,
                "screen": None,
                "gate": None,
                "effect": None,
                "decision": "REJECTED",
            }
            try:
                validate_mutation(mutation)
                if mutation["target_cluster_id"] != cluster["cluster_id"]:
                    raise ValueError("proposal changed its target cluster")
                if not set(
                    mutation["evidence_task_ids"]
                    + mutation["protected_success_task_ids"]
                ) <= set(e):
                    raise ValueError("proposal names a task outside E")
                proposed = apply_v2_mutation(
                    before_s, mutation, next_skill_id_number=watermark + 1
                )
                watermark = max(watermark, service_skill_id_high_watermark(proposed))
                validate_skill_budgets(proposed, policy["skill_budgets"])
                row["proposed_memory"] = proposed.to_dict()
            except (TypeError, ValueError, KeyError) as exc:
                row["pre_rollout_rejection"] = f"invalid_mutation: {exc}"
                return row, ()
            if proposed == before_s:
                row["pre_rollout_rejection"] = "no_op"
                return row, ()
            duplicate = (
                semantic_duplicate(mutation, [*effects, *seen_candidates])
                if evo["semantic_dedup"]
                else None
            )
            if duplicate:
                row.update(
                    pre_rollout_rejection="semantic_duplicate", duplicate_of=duplicate
                )
                return row, ()
            validation_context = {
                "skills": proposed.to_dict(),
                "native_policy": domain_policy,
            }
            if analyst_mode:
                validation_context["algorithm_version"] = policy["algorithm_version"]
            validation = stage(
                g,
                f"skill-validator-{ident}",
                validation_context,
                lambda: providers.validate_skill(validation_context),
            )
            row["semantic_validation"] = validation
            if not all(
                validation.get(k) is True
                for k in ("reusable", "policy_subordinate", "no_task_entities")
            ):
                row["pre_rollout_rejection"] = "skill_semantics_failed"
                return row, ()
            prior_fixed = {
                t for ent in effects for t in ent["effect"]["fail_to_pass"] if t in e
            }
            clean = [t for t in e if t not in target | protected | prior_fixed]
            random.Random(seed + g).shuffle(clean)
            screen_ids = tuple(
                t
                for t in e
                if t
                in target
                | protected
                | prior_fixed
                | set(clean[: policy["evaluation"]["screen_clean_tasks"]])
            )
            screen_seeds = policy["evaluation"]["screen_seeds"]
            old = panel(
                g,
                f"service_screen-{ident}-old",
                screen_ids,
                screen_seeds,
                customer,
                before_s,
            )
            new = panel(
                g,
                f"service_screen-{ident}-new",
                screen_ids,
                screen_seeds,
                customer,
                proposed,
            )
            screen = stage(
                g,
                f"service_screen-{ident}-decision",
                {
                    "old": [r.to_dict() for r in old],
                    "new": [r.to_dict() for r in new],
                    "target": sorted(target),
                    "protected": sorted(protected | prior_fixed),
                },
                lambda: cheap_screen(old, new, target, protected | prior_fixed,
                    max_regression_rate=policy["evaluation"]["screen_max_regression_rate"] if v_primary else 0.0,
                    regression_allowance=policy["evaluation"]["screen_regression_allowance"] if v_primary else 0,
                    max_stuck_delta=policy["evaluation"]["screen_max_stuck_delta"] if v_primary else 0.0),
            )
            target_decisions = []
            if frozen_targets:
                from .targeted_repair_gate import evaluate_target
                for d in frozen_targets:
                    challenge = PromptStrategy(d["compiled_customer"]["text"])
                    tids, tseeds = [d["discovery"]["task_id"]], d["discovery"]["seeds"]
                    target_cost_before = stage(g, f"target-{ident}-{d['discovery_id']}-cost-start",
                        {"discovery_id": d["discovery_id"], "policy": targeted_policy}, rc.costs)
                    base = panel(g, f"target-{ident}-{d['discovery_id']}-old", tids, tseeds, challenge, before_s)
                    candidate = panel(g, f"target-{ident}-{d['discovery_id']}-new", tids, tseeds, challenge, proposed)
                    decision = stage(g, f"target-{ident}-{d['discovery_id']}-decision",
                        {"old": [r.to_dict() for r in base], "new": [r.to_dict() for r in candidate],
                         "discovery": d, "policy": targeted_policy},
                        lambda base=base, candidate=candidate, d=d, target_cost_before=target_cost_before: {
                            **evaluate_target(base, candidate, d, before_s, proposed, targeted_policy),
                            "execution_cost": {key: None if value is None or target_cost_before.get(key) is None
                                               else value - target_cost_before[key] for key, value in rc.costs().items()}})
                    target_decisions.append(decision)
            if targeted_policy:
                row["targeted_repair"] = target_decisions
            if frozen_targets:
                # Initial E screen protects incumbent successes; target gate owns improvement.
                protection = evaluate_gate(old, new, policy["statistical_gate"], objective="preservation", smoke=True, seed=seed)
                screen = {**screen, "passed": protection["verdict"] == "ACCEPTED" and all(t["verdict"] == "ACCEPTED" for t in target_decisions),
                          "reason": "target improvement and observed incumbent protection", "incumbent_protection": protection}
            row["screen"] = screen
            attribution_old, attribution_new = old, new
            gates, full_runs = [], ()
            if screen["passed"]:
                if not smoke:
                    attribution_old = panel(
                        g,
                        f"service_repair_full-{ident}-old",
                        e,
                        evaluation_seed_schedule(policy, "E"),
                        customer,
                        before_s,
                    )
                    attribution_new = panel(
                        g,
                        f"service_repair_full-{ident}-new",
                        e,
                        evaluation_seed_schedule(policy, "E"),
                        customer,
                        proposed,
                    )
                if frozen_targets:
                    # All incumbent/native/replay E successes are protected, not just target tasks.
                    for label, opponent, _ in opponents:
                        protected_old = panel(g, f"target-protection-{ident}-{label}-old", e,
                                              evaluation_seed_schedule(policy, "E"), opponent, before_s)
                        protected_new = panel(g, f"target-protection-{ident}-{label}-new", e,
                                              evaluation_seed_schedule(policy, "E"), opponent, proposed)
                        protection = stage(g, f"target-protection-{ident}-{label}-decision",
                            {"old": [r.to_dict() for r in protected_old], "new": [r.to_dict() for r in protected_new],
                             "policy": policy["statistical_gate"]},
                            lambda protected_old=protected_old, protected_new=protected_new: evaluate_gate(
                                protected_old, protected_new, policy["statistical_gate"], objective="preservation", smoke=True, seed=seed))
                        protection["opponent"] = label + "-protected-E"
                        target_decisions.append(protection)
                gate_ids = e if smoke else v
                gate_seeds = policy["evaluation"]["gate_seeds"]
                repair_gate = None
                if not smoke and not v_primary and not frozen_targets:
                    repair_gate = stage(
                        g,
                        f"service_full_gate-{ident}-repair-superiority-decision",
                        {
                            "old": [r.to_dict() for r in attribution_old],
                            "new": [r.to_dict() for r in attribution_new],
                            "policy": policy["statistical_gate"],
                            "objective": "superiority",
                            "smoke": True,
                            "looks": looks,
                        },
                        lambda: evaluate_gate(
                            attribution_old,
                            attribution_new,
                            policy["statistical_gate"],
                            objective="superiority",
                            looks=looks,
                            smoke=True,
                            seed=seed,
                        ),
                    )
                    repair_gate.update(
                        opponent="current-repair-E",
                        pass_power_k={
                            str(k): pass_power_k(attribution_new, k) for k in (1, 2)
                        },
                    )
                for opponent_index, (label, opponent, weight) in enumerate(
                    opponents
                    if repair_gate is None or repair_gate["verdict"] == "ACCEPTED"
                    else []
                ):
                    objective = (
                        "superiority"
                        if (smoke or v_primary) and label == "current" and not frozen_targets
                        else "preservation"
                    )
                    old_gate = panel(
                        g,
                        f"service_full_gate-{ident}-{label}-old",
                        gate_ids,
                        gate_seeds,
                        opponent,
                        before_s,
                    )
                    new_gate = panel(
                        g,
                        f"service_full_gate-{ident}-{label}-new",
                        gate_ids,
                        gate_seeds,
                        opponent,
                        proposed,
                    )
                    args = {
                        "old": [r.to_dict() for r in old_gate],
                        "new": [r.to_dict() for r in new_gate],
                        "policy": policy["statistical_gate"],
                        "looks": looks,
                        "smoke": smoke,
                        "objective": objective,
                    }
                    gate = stage(
                        g,
                        f"service_full_gate-{ident}-{label}-decision",
                        args,
                        lambda baseline=old_gate, proposed_runs=new_gate, objective=objective: (
                            evaluate_gate(
                                baseline,
                                proposed_runs,
                                policy["statistical_gate"],
                                looks=looks,
                                smoke=smoke,
                                seed=seed,
                                objective=objective,
                            )
                        ),
                    )
                    gate.update(
                        opponent=label,
                        opponent_weight=weight,
                        pass_power_k={
                            str(k): pass_power_k(new_gate, k) for k in (1, 2)
                        },
                    )
                    gates.append(gate)
                    if label == "current":
                        full_runs = new_gate
                        if smoke:
                            attribution_old, attribution_new = old_gate, new_gate
                    if (
                        not smoke
                        and gate["verdict"] == "REJECTED"
                        and policy["evaluation"]["v_gate_mode"] == "fail_fast"
                    ):
                        # The conjunction cannot pass. Preserve frozen looks and explicit gaps.
                        skipped = stage(
                            g,
                            f"service_full_gate-{ident}-remaining-not-evaluated",
                            {
                                "rejected_opponent": label,
                                "remaining": [
                                    x[0] for x in opponents[opponent_index + 1 :]
                                ],
                                "looks": looks,
                                "panel": "V",
                                "seeds": gate_seeds,
                            },
                            lambda remaining=opponents[opponent_index + 1 :]: {
                                "opponents": [
                                    {
                                        "opponent": later,
                                        "opponent_weight": later_weight,
                                        "verdict": "NOT_EVALUATED",
                                        "objective": "preservation",
                                        "reason": "Earlier required V gate rejected; incomplete risk profile",
                                        "panel": "V",
                                        "seeds": gate_seeds,
                                        "gate_looks": looks,
                                        "policy": policy["statistical_gate"],
                                        "pass_power_k": {},
                                    }
                                    for later, _, later_weight in remaining
                                ]
                            },
                        )
                        gates.extend(skipped["opponents"])
                        break
                required_gates = gates + target_decisions + (
                    [repair_gate] if repair_gate is not None else []
                )
                verdict = (
                    "REJECTED"
                    if any(x["verdict"] == "REJECTED" for x in required_gates)
                    else "INCONCLUSIVE"
                    if any(x["verdict"] == "INCONCLUSIVE" for x in required_gates)
                    else "ACCEPTED"
                )
                row["gate"] = {
                    "verdict": verdict,
                    "opponents": gates,
                    "repair_superiority": repair_gate,
                    "gate_looks": looks,
                    "promotion_protocol": targeted_policy["protocol_version"] if frozen_targets else policy["evaluation"].get("promotion_protocol", "legacy_e_superiority"),
                    "evaluation_mode": policy["evaluation"]["v_gate_mode"],
                    "risk_profile_complete": len(gates) == len(opponents)
                    and all(x["verdict"] != "NOT_EVALUATED" for x in gates),
                    "reason": "; ".join(
                        x.get("opponent", "discovery-target-E") + ": " + x["reason"] for x in required_gates
                    ),
                }
            else:
                verdict = ("INCONCLUSIVE" if frozen_targets and any(t["verdict"] == "INCONCLUSIVE" for t in target_decisions)
                           and not any(t["verdict"] == "REJECTED" for t in target_decisions) else "REJECTED")
            reason = row["gate"]["reason"] if row["gate"] else screen["reason"]
            effect = paired_effect(
                attribution_old,
                attribution_new,
                mutation,
                generation=g,
                mutation_id=ident,
                parent_memory_id=service_strategy_id(before_s),
                proposed_memory_id=service_strategy_id(proposed),
                reason=reason,
                parent_ids=parents,
            ).to_dict()
            row.update(
                effect=effect,
                decision=verdict,
                failure_matrix=paired_failure_matrix(
                    attribution_old,
                    attribution_new,
                    generation=g,
                    candidate_id=ident,
                    prior=matrix,
                ),
                evaluation_scope="full_E" if screen["passed"] else "paired_screen_E",
                full_gate_episodes=[r.to_dict() for r in full_runs],
                comparison_key=sha256_json(
                    {
                        "cells": sorted((r.task_id, r.seed) for r in attribution_old),
                        "customer_ids": sorted(
                            {r.customer_strategy_id for r in attribution_old}
                        ),
                        "service_ids": sorted(
                            {r.service_strategy_id for r in attribution_old}
                        ),
                    }
                ),
            )
            return row, full_runs

        evaluated = list(rejected_proposals)
        for ident, mutation, cluster, parents in all_proposals:
            row, runs = evaluate_candidate(ident, mutation, cluster, parents)
            evaluated.append((row, runs))
            if isinstance(row.get("mutation"), dict) and row["mutation"].get(
                "operation"
            ) in (
                "add",
                "narrow_trigger",
                "expand_trigger",
                "rewrite_guidance",
                "split",
                "delete",
                "no_op",
            ):
                seen_candidates.append(row)
        if evo["crossover"] and (not analyst_mode or assignments):
            eligible = [
                row
                for row, _ in evaluated
                if row["effect"] and row["effect"]["fail_to_pass"]
            ]
            eligible.extend(archive.entries)
            pair = next(
                (
                    (a, b)
                    for i, a in enumerate(eligible)
                    for b in eligible[i + 1 :]
                    if a["mutation_id"] != b["mutation_id"] and crossover_eligible(a, b)
                ),
                None,
            )
            if pair:
                a, b = pair
                ctx = {
                    **s_context,
                    "parents": [prompt_parent(a), prompt_parent(b)],
                    "current_memory": before_s.to_dict(),
                    "policy": domain_policy,
                }
                from .evolution_candidates import (
                    CROSSOVER_PROMPT,
                    DIRECT_MUTATOR_PROMPT,
                )

                enforce_mutation_context_budget(
                    ctx,
                    CROSSOVER_PROMPT + DIRECT_MUTATOR_PROMPT,
                    policy["mutation_context"]["max_proxy_tokens"],
                )
                parents = (a["mutation_id"], b["mutation_id"])
                if (
                    evo.get("candidate_error_policy", "fail_closed")
                    == "reject_candidate"
                ):
                    outcome = stage(
                        g,
                        "service_crossover",
                        ctx,
                        lambda ctx=ctx: _candidate_outcome(
                            lambda: providers.crossover(ctx), ctx
                        ),
                    )
                else:
                    mutation = stage(
                        g,
                        "service_crossover",
                        ctx,
                        lambda ctx=ctx: providers.crossover(ctx),
                    )
                    validate_mutation(mutation)
                    validate_direct_evidence(mutation, ctx)
                    outcome = {"status": "VALID", "mutation": mutation}
                if outcome["status"] == "REJECTED":
                    evaluated.append(
                        (
                            _rejected_candidate(
                                f"g{g:04d}-crossover", outcome, g, parents
                            ),
                            (),
                        )
                    )
                else:
                    mutation = outcome["mutation"]
                    row, runs = evaluate_candidate(
                        f"g{g:04d}-crossover",
                        mutation,
                        candidate_hypothesis(mutation),
                        parents,
                    )
                    evaluated.append((row, runs))

        qualified = [
            (row, runs) for row, runs in evaluated if row["decision"] == "ACCEPTED"
        ]

        def quality(pair):
            row = pair[0]
            current_gate = (
                row["gate"].get("repair_superiority") or row["gate"]["opponents"][0]
            )
            return (
                -current_gate["new_metrics"]["accuracy"],
                row["effect"]["harmfulness"] or 0,
                row["mutation_id"],
            )

        winner = min(qualified, key=quality) if qualified else None
        for row, runs in evaluated:
            accepted = (
                winner is not None and row["mutation_id"] == winner[0]["mutation_id"]
            )
            row["runtime_deployed"] = accepted
            if row["effect"]:
                row["effect"]["accepted"] = accepted
                if row["decision"] == "ACCEPTED" and not accepted:
                    row["decision"] = "QUALIFIED_NOT_SELECTED"
                    row["effect"]["rejection_reason"] = (
                        "another qualified candidate selected deterministically"
                    )
                if accepted:
                    row["effect"]["rejection_reason"] = None
                effects.append(row)
                matrix.extend(row["failure_matrix"])
                if policy["archive"]["enabled"]:
                    archive.add(row)
            board.append(row)
        if discovery_archive:
            for candidate_row in board:
                for d in frozen_targets:
                    discovery_archive.event(d["discovery_id"], g, candidate_row["mutation_id"],
                        {"decision": candidate_row["decision"], "deployed": candidate_row["runtime_deployed"],
                         "target_evidence": candidate_row.get("targeted_repair"),
                         "gate_artifact_sha256": sha256_json(candidate_row.get("gate")),
                         "evaluation_ref": f"generation-{g:04d}.json"})
        if winner:
            row, _ = winner
            service = ServiceSkillMemoryV2.from_mapping(row["proposed_memory"])
            live_ids = {s.skill_id for s in service.skills}
            prior_provenance = {p["skill_id"]: p for p in provenance}
            provenance = [
                prior_provenance[s.skill_id]
                if s in before_s.skills
                else {
                    "skill_id": s.skill_id,
                    "created_generation": prior_provenance.get(s.skill_id, {}).get(
                        "created_generation", g
                    ),
                    "updated_generation": g,
                    "parent_mutation": row["mutation_id"],
                    "source_task_ids": row["effect"]["fail_to_pass"],
                }
                for s in service.skills
                if s.skill_id in live_ids
            ]
            final_runs = panel(
                g, "service-selected-final-E", e, [fitness_seed], customer, service
            )
        else:
            final_runs = selected
        stage(
            g,
            "service_selection",
            {"board": board, "before": before_s.to_dict()},
            lambda service=service, winner=winner: {
                "service": service.to_dict(),
                "accepted": winner is not None,
            },
        )
        stage(
            g,
            "archive_update",
            {"effects": effects, "customer_archive": customers.entries},
            lambda archive=archive, effects=effects: {
                "archive": archive.to_dict(),
                "history": summarize_effects(effects),
            },
        )
        if rc:
            from .repair_feedback import service_pair
            # Only the completed previous generation may inform the next Customer.
            # No promotion resets the frontier rather than fabricating a repair.
            repair_before, repair_after = tuple(selected), tuple(final_runs)
            if discovery_archive and winner:
                # Keep the actual promoted target's training repair, not only incumbent E.
                for decision in winner[0].get("targeted_repair", []) or []:
                    if decision.get("discovery_id"):
                        repair_before += tuple(EpisodeRecord.from_dict(r) for r in decision["old"])
                        repair_after += tuple(EpisodeRecord.from_dict(r) for r in decision["new"])
            repair_pair = service_pair(before_s, service, generation=g, promoted=winner is not None,
                                       before_records=repair_before, after_records=repair_after)
        if policy.get("open_repair_search"):
            rc.repair_evidence = None if repair_pair is None else {
                endpoint_name: build_service_mutation_evidence(service_evidence(records),
                    representative_cases=customer_policy["representative_cases"], case_chars=customer_policy["case_chars"])
                for endpoint_name, records in (("before", repair_before), ("after", repair_after))}
        # Console backward-compatible common phase fields plus explicit V2 board.
        proposed_accuracy = (
            winner[0]["effect"]["candidate_accuracy"]
            if winner
            else next(
                (r["effect"]["candidate_accuracy"] for r in board if r["effect"]), None
            )
        )
        from .evolution_artifacts import summarize_activation_artifacts

        activation_summary = summarize_activation_artifacts(
            root, manifest_sha256, set(e) | set(v)
        )
        gen = {
            "schema_version": 3,
            "manifest_sha256": manifest_sha256,
            "generation": g,
            "evolution_fitness_seed": fitness_seed,
            "customer_before": _document("customer", before_c),
            "service_before": _document("service", before_s),
            "customer_after": _document("customer", customer),
            "service_after": _document("service", service),
            "customer_phase": {
                "incumbent_accuracy": _accuracy(incumbent),
                "candidate_accuracies": [r.get("accuracy") for r in candidate_rows],
                "selected_accuracy": selected_accuracy,
                "selected_customer": selected_source,
                "candidates": candidate_rows,
                "incumbent_episodes": [r.to_dict() for r in incumbent],
            },
            "service_phase": {
                **({"failure_analysis": analyst_outcome, "hypothesis_selection": hypothesis_selection} if analyst_mode else {}),
                "carrier": "skill_memory_v2",
                "validation_looks_evaluated": sum(
                    gate["verdict"] != "NOT_EVALUATED"
                    for row in board for gate in (row.get("gate") or {}).get("opponents", [])
                    if gate.get("panel") == "V"
                ),
                "old_accuracy": selected_accuracy,
                "final_accuracy": _accuracy(final_runs),
                "proposed_accuracy": proposed_accuracy,
                "accepted": winner is not None,
                "acceptance_mode": "e_only_mechanism_smoke"
                if smoke
                else "finite_panel_validation_gated"
                if policy["statistical_gate"]["method"] == "finite_panel_paired"
                else "statistical_validation_gated",
                "validation_evaluated": any(
                    (r["gate"] or {}).get("opponents") for r in board
                )
                and not smoke,
                "selection": {
                    "reason": "qualified candidate promoted"
                    if winner
                    else "no candidate passed all screen/promotion/replay gates"
                },
                "skill_count_before": len(before_s.skills),
                "skill_count_after": len(service.skills),
                "candidates": board,
                "algorithm_version": policy["algorithm_version"],
                "candidate_budget": evo["candidates_per_generation"],
                "crossover_budget": int(evo["crossover"]),
                "opponents_planned": [label for label, _, _ in opponents],
                "opponents_replayed": [
                    label
                    for label, _, _ in opponents
                    if any(
                        gate["opponent"] == label and gate["verdict"] != "NOT_EVALUATED"
                        for row in board
                        for gate in (row.get("gate") or {}).get("opponents", [])
                    )
                ],
                "exploration": s_context["exploration"],
            },
            "evolution_health": {
                "active_skill_count": len(service.skills),
                "archive_size": len(archive.entries),
                "activation_mode": policy["service_skill_runtime"],
                "mechanism_smoke": smoke,
            },
            "activation_summary": activation_summary,
            "customer_archive": customers.entries,
            **({"customer_protocol": PROTOCOL, "customer_attempt_history": customer_attempts.entries} if customer_policy else {}),
            **({"open_repair_evidence": rc.repair_evidence} if policy.get("open_repair_search") else {}),
            **({"bandit_protocol": RC_PROTOCOL, "bandit_state": rc.state,
                "repair_service_pair": repair_pair} if rc else {}),
            "failure_matrix": [
                r for row in board for r in row.get("failure_matrix", [])
            ],
            "history_summary": summarize_effects(effects),
            "statistical_policy": policy["statistical_gate"],
            "timing": {
                "validation_wall_clock_seconds": sum(
                    elapsed
                    for key, elapsed in timing["stages"].items()
                    if key.startswith(f"g{g:04d}-service_full_gate")
                    and "repair-superiority-decision" not in key
                    and not smoke
                )
            },
        }
        documents.append(deepcopy(gen))
        state = {
            "customer": customer.to_dict(),
            "service": service.to_dict(),
            "service_provenance": provenance,
            "skill_id_high_watermark": watermark,
            "mutation_effects": effects,
            "failure_matrix": matrix,
            "evolution_archive": archive.to_dict(),
            "customer_archive": customers.entries,
            **({"customer_protocol": PROTOCOL, "customer_attempt_history": customer_attempts.entries} if customer_policy else {}),
            **({"open_repair_evidence": rc.repair_evidence} if policy.get("open_repair_search") else {}),
            **({"bandit_protocol": RC_PROTOCOL, "bandit_state": rc.state,
                "repair_service_pair": repair_pair} if rc else {}),
            "history_summary": summarize_effects(effects),
            "activation_config": policy["activator"],
            "activation_summary": activation_summary,
            "statistical_gate_state": {
                "policy": policy["statistical_gate"],
                "looks": looks,
            },
            "final_evolution_episodes": [r.to_dict() for r in final_runs],
        }
        stage(
            g,
            "generation_complete",
            frozen_start,
            lambda gen=gen, state=state: {"generation": gen, "state": state},
        )
        _write_json_once(root / f"generation-{g:04d}.json", gen)
        checkpoint = {
            "schema_version": 3,
            "manifest_sha256": manifest_sha256,
            "completed_generation": g,
            "generations": documents,
            "state": state,
        }
        checkpoint["checkpoint_sha256"] = sha256_json(checkpoint)
        _write_json_atomic(checkpoint_file, checkpoint)
    # Reconstruct checkpoint after a crash between immutable commit and atomic checkpoint.
    last = journal.read(
        journal.root / f"g{generations - 1:04d}-generation_complete.json"
    )["payload"]["state"]
    checkpoint = {
        "schema_version": 3,
        "manifest_sha256": manifest_sha256,
        "completed_generation": generations - 1,
        "generations": documents,
        "state": last,
    }
    checkpoint["checkpoint_sha256"] = sha256_json(checkpoint)
    _write_json_atomic(checkpoint_file, checkpoint)
    from .service_skills import ServiceSkillProvenance

    compat_provenance = tuple(
        ServiceSkillProvenance(
            p["skill_id"],
            p["created_generation"],
            p["updated_generation"],
            tuple(p["source_task_ids"]),
        )
        for p in provenance
    )
    return AlternatingResult(
        initial_customer,
        initial_service,
        customer,
        service,
        tuple(documents),
        tuple(final_runs),
        compat_provenance,
    )


def propose_fresh_customer_v2(
    result,
    providers,
    *,
    tasks,
    runner,
    domain_policy,
    output_directory,
    manifest_sha256,
    mutation_context=None,
    customer_policy=None,
    max_parallel_episodes=1,
):
    """Freeze and validate the final adaptive challenge using E-only evidence before H loads."""
    from .alternating import _context_episodes, _write_json_once
    from .evolution_context import (
        CUSTOMER_EVIDENCE_INSTRUCTIONS,
        build_customer_evidence,
    )

    if customer_policy:
        from .customer_evolution import propose_fresh_customer_skill
        return propose_fresh_customer_skill(result, providers, tasks=tasks, runner=runner,
            domain_policy=domain_policy, output_directory=output_directory,
            manifest_sha256=manifest_sha256, policy=customer_policy,
            max_parallel_episodes=max_parallel_episodes)
    journal = EvolutionJournal(output_directory, manifest_sha256)
    context = {
        "task_interactions": build_customer_evidence(
            _context_episodes(result.final_evolution_episodes, runner, tasks),
            representative_cases=(mutation_context or {}).get("representative_cases", 4),
            case_chars=(mutation_context or {}).get("case_chars", 12000),
        ),
        "evidence_instructions": CUSTOMER_EVIDENCE_INSTRUCTIONS,
        "current_customer_strategy": result.customer.text,
        "current_service_memory": result.service.to_dict(),
        "service_policy": domain_policy,
        "challenge_archive": result.generations[-1].get("customer_archive", []),
        "purpose": "Fresh adaptive challenge; heldout content has not been loaded.",
    }
    proposal = journal.freeze(
        "final-fresh-customer", context, lambda: providers.customers(context, 1)
    )["candidates"][0]
    validation_context = {
        "strategy": proposal["strategy"],
        "scenarios": [
            row["task"]["user_scenario"] for row in context["task_interactions"]
        ],
    }
    validation = journal.freeze(
        "final-fresh-customer-validator",
        validation_context,
        lambda: providers.validate_customer(validation_context),
    )
    valid = all(
        validation.get(key) is True
        for key in (
            "preserves_facts",
            "preserves_objective",
            "interaction_only",
            "no_benchmark_leakage",
        )
    )
    # One predeclared fresh attempt. Rejection is research evidence, not an exception.
    # Do not relabel an incumbent/archive Customer as a fresh challenge.
    distinct = proposal["strategy"].strip() not in {
        result.customer.text.strip(),
        result.initial_customer.text.strip(),
        *(entry["strategy"].strip() for entry in context["challenge_archive"]),
    }
    customer = PromptStrategy(proposal["strategy"]) if valid and distinct else None
    _write_json_once(
        Path(output_directory) / "fresh-customer-proposal.json",
        {
            "schema_version": 3,
            "manifest_sha256": manifest_sha256,
            "evolver_input_sha256": sha256_json(context),
            "status": "available" if customer is not None else "unavailable",
            "selection_protocol": "one E-only proposal; semantic preservation and distinct text required; otherwise native H only",
            "rejection_reason": None
            if customer is not None
            else (
                "semantic_preservation_failed"
                if not valid
                else "not_distinct_from_evolution_customers"
            ),
            "strategy": proposal["strategy"],
            "strategy_id": customer_strategy_id(PromptStrategy(proposal["strategy"])),
            "semantic_validation": validation,
        },
    )
    return customer


def run_v2_endpoint_evaluation(*, gate_seeds, **kwargs):
    """Final-only native/fresh H scorecard; reuse identical S0/ST conditions for each seed."""
    from .alternating import _write_json_atomic, run_final_endpoint_evaluation
    from .evolution_gate import pass_power_k

    output_path = kwargs.pop("output_path", None)
    kwargs.pop("seed", None)
    per_seed = [
        run_final_endpoint_evaluation(seed=seed, **kwargs) for seed in gate_seeds
    ]
    summary = deepcopy(per_seed[0])
    summary.update(schema_version=3, seeds=list(gate_seeds), per_seed=per_seed)
    cells = []
    for index in range(len(per_seed[0]["cells"])):
        cell = deepcopy(per_seed[0]["cells"][index])
        episodes = [
            e for result in per_seed for e in result["cells"][index]["episodes"]
        ]
        records = [EpisodeRecord.from_dict(e) for e in episodes]
        cell.update(
            accuracy=episode_metrics(records)["accuracy"],
            metrics=episode_metrics(records),
            episodes=episodes,
            pass_power_k={str(k): pass_power_k(records, k) for k in (1, 2, 4)},
        )
        cells.append(cell)
    summary["cells"] = cells
    comparisons = []
    for condition in sorted({c["customer_condition"] for c in cells}):
        before = next(c for c in cells if c["customer_condition"] == condition and c["service_endpoint"] == "S0")
        after = next(c for c in cells if c["customer_condition"] == condition and c["service_endpoint"] == "ST")
        a = {(r["task_id"], r["seed"]): r for r in before["episodes"]}
        b = {(r["task_id"], r["seed"]): r for r in after["episodes"]}
        if a.keys() != b.keys():
            raise ValueError("H scorecard requires paired endpoint cells")
        comparisons.append({"customer_condition": condition,
            "identical_to_S0": after["identical_to_S0"],
            "fail_to_pass_count": sum(a[k]["task_success"] is False and b[k]["task_success"] is True for k in a),
            "pass_to_fail_count": sum(a[k]["task_success"] is True and b[k]["task_success"] is False for k in a),
            "causal_skill_effect_identified": False})
    summary["endpoint_comparisons"] = comparisons
    if output_path:
        _write_json_atomic(Path(output_path), summary)
    return summary
