"""V2 orchestration: diagnose/diversify/screen/gate, isolated from legacy V1."""

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
from .evolution_candidates import validate_mutation
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

    root = Path(output_directory)
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
                return callback()
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
            documents.append(commit["generation"])
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
            else effects
        )
        c_context = {
            "generation": g,
            "task_interactions": _context_episodes(incumbent, runner, tasks),
            "current_customer_strategy": customer.text,
            "current_service": service.to_dict(),
            "incumbent_accuracy": _accuracy(incumbent),
            "service_policy": domain_policy,
            "challenge_archive": customers.entries,
        }
        proposals = stage(
            g,
            "customer_candidates",
            c_context,
            lambda c_context=c_context: providers.customers(
                c_context, customer_candidate_count
            ),
        )["candidates"]
        candidate_rows, selected, selected_accuracy = (
            [],
            incumbent,
            _accuracy(incumbent),
        )
        selected_source = "incumbent"
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
        # Generate diagnoses only from E; passing and failing examples are equally available.
        all_evidence = service_evidence(selected)
        if policy["history"]["summarize"]:
            limit = policy["history"]["representative_cases"]
            failures = [
                row
                for row in all_evidence
                if row["native_evaluation"]["task_success"] is False
            ]
            passing = [
                row
                for row in all_evidence
                if row["native_evaluation"]["task_success"] is True
            ]

            def visible_tools(row):
                return {
                    tool.get("name") or (tool.get("function") or {}).get("name")
                    for message in (row.get("trajectory") or {}).get("messages", [])
                    for tool in (message.get("tool_calls") or [])
                }

            failure_tools = set().union(*(visible_tools(row) for row in failures))
            # Similar visible tool paths are controls, never hidden objectives/task identities.
            passing.sort(key=lambda row: -len(visible_tools(row) & failure_tools))
            # Keep failures and passing controls; IDs refer to the full native E panel.
            representative = (
                failures[: max(1, limit // 2)]
                + passing[: max(1, limit - len(failures[: max(1, limit // 2)]))]
            )
        else:
            representative = all_evidence
        s_context = {
            "generation": g,
            "task_interactions": representative,
            "failure_matrix": [r for r in matrix if r["task_id"] in e][-len(e) * 8 :],
            "current_service_memory": before_s.to_dict(),
            "service_policy": domain_policy,
            "interaction_metrics": episode_metrics(selected),
            "current_outcomes": [r.to_dict() for r in selected],
            "previous_mutation_effects": history if evo["lineage"] else [],
            "exploration": stagnation_state(
                documents, effects, evo["stagnation_patience"]
            ),
        }
        diagnosis = stage(
            g,
            "service_diagnosis",
            s_context,
            lambda s_context=s_context: providers.diagnose(s_context),
        )
        failed_ids = {r.task_id for r in selected if r.task_success is False}
        passed_ids = {r.task_id for r in selected if r.task_success is True}
        for cluster in diagnosis["clusters"]:
            if (
                not set(cluster["evidence_task_ids"]) <= failed_ids
                or not set(cluster["protected_success_task_ids"]) <= passed_ids
            ):
                raise ValueError(
                    "diagnosis target/protected labels disagree with native E evidence"
                )
        clusters = [
            c for c in diagnosis["clusters"] if c["recommended_surface"] == "skill"
        ]
        if s_context["exploration"]["explore"]:
            used = {entry["mutation"]["target_cluster_id"] for entry in effects[-3:]}
            clusters.sort(key=lambda c: (c["cluster_id"] in used, c["cluster_id"]))
        clusters = clusters[: evo["max_clusters_per_generation"]]
        board = []
        looks = evo["candidates_per_cluster"] * evo[
            "max_clusters_per_generation"
        ] + int(evo["crossover"])
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
        looks *= len(opponents)
        all_proposals = []
        for cluster in clusters:
            allowed = set(e)
            target, protected = (
                set(cluster["evidence_task_ids"]),
                set(cluster["protected_success_task_ids"]),
            )
            if not target <= allowed or not protected <= allowed:
                raise ValueError("repair clusters must use E evidence only")
            for bias in range(evo["candidates_per_cluster"]):
                ctx = {
                    **s_context,
                    "root_cause_cluster": cluster,
                    "proposal_bias": (
                        "narrow_applicability",
                        "minimal_behavior",
                        "structural_decomposition",
                    )[bias % 3],
                    "prior_fixed_cases": sorted(
                        {
                            t
                            for ent in effects
                            for t in ent["effect"]["fail_to_pass"]
                            if t in e
                        }
                    ),
                    "prior_broken_cases": sorted(
                        {
                            t
                            for ent in effects
                            for t in ent["effect"]["pass_to_fail"]
                            if t in e
                        }
                    ),
                    "archive_parent": (
                        archive.entries[bias % len(archive.entries)]
                        if archive.entries and s_context["exploration"]["explore"]
                        else None
                    ),
                }
                relevant_ids = (
                    target
                    | protected
                    | set(ctx["prior_fixed_cases"])
                    | set(ctx["prior_broken_cases"])
                )
                relevant = [
                    row
                    for row in all_evidence
                    if row["task"]["task_id"] in relevant_ids
                ]
                controls = [
                    row
                    for row in representative
                    if row["task"]["task_id"] not in relevant_ids
                ]
                ctx["task_interactions"] = (
                    relevant
                    + controls[
                        : max(
                            0, policy["history"]["representative_cases"] - len(relevant)
                        )
                    ]
                )
                ident = f"g{g:04d}-{cluster['cluster_id']}-{bias}"
                proposal = stage(
                    g,
                    f"service_proposals-{ident}",
                    ctx,
                    lambda ctx=ctx: providers.mutate(ctx),
                )
                all_proposals.append((ident, proposal, cluster, ()))

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
        ):
            nonlocal watermark
            target, protected = (
                set(cluster["evidence_task_ids"]),
                set(cluster["protected_success_task_ids"]),
            )
            row = {
                "mutation_id": ident,
                "mutation": mutation,
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
                    mutation["expected_fixes"] + mutation["protected_cases_at_risk"]
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
                lambda: cheap_screen(old, new, target, protected | prior_fixed),
            )
            row["screen"] = screen
            attribution_old, attribution_new = old, new
            gates, full_runs = [], ()
            if screen["passed"]:
                if not smoke:
                    attribution_old = panel(
                        g,
                        f"service_repair_full-{ident}-old",
                        e,
                        policy["evaluation"]["gate_seeds"],
                        customer,
                        before_s,
                    )
                    attribution_new = panel(
                        g,
                        f"service_repair_full-{ident}-new",
                        e,
                        policy["evaluation"]["gate_seeds"],
                        customer,
                        proposed,
                    )
                gate_ids = e if smoke else v
                gate_seeds = policy["evaluation"]["gate_seeds"]
                for label, opponent, weight in opponents:
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
                    }
                    gate = stage(
                        g,
                        f"service_full_gate-{ident}-{label}-decision",
                        args,
                        lambda baseline=old_gate, proposed_runs=new_gate: evaluate_gate(
                            baseline,
                            proposed_runs,
                            policy["statistical_gate"],
                            looks=looks,
                            smoke=smoke,
                            seed=seed,
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
                verdict = (
                    "REJECTED"
                    if any(x["verdict"] == "REJECTED" for x in gates)
                    else "INCONCLUSIVE"
                    if any(x["verdict"] == "INCONCLUSIVE" for x in gates)
                    else "ACCEPTED"
                )
                row["gate"] = {
                    "verdict": verdict,
                    "opponents": gates,
                    "gate_looks": looks,
                    "reason": "; ".join(
                        x["opponent"] + ": " + x["reason"] for x in gates
                    ),
                }
            else:
                verdict = "REJECTED"
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
            )
            return row, full_runs

        evaluated = []
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
        if evo["crossover"]:
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
                cluster = (
                    clusters[0]
                    if clusters
                    else {
                        "cluster_id": "crossover",
                        "evidence_task_ids": sorted(
                            set(a["effect"]["fail_to_pass"])
                            | set(b["effect"]["fail_to_pass"])
                        ),
                        "protected_success_task_ids": sorted(
                            set(a["effect"]["pass_to_pass"])
                            | set(b["effect"]["pass_to_pass"])
                        ),
                    }
                )
                ctx = {
                    "parents": [a, b],
                    "root_cause_cluster": cluster,
                    "current_memory": before_s.to_dict(),
                    "policy": domain_policy,
                }
                mutation = stage(
                    g,
                    "service_crossover",
                    ctx,
                    lambda ctx=ctx: providers.crossover(ctx),
                )
                row, runs = evaluate_candidate(
                    f"g{g:04d}-crossover",
                    mutation,
                    cluster,
                    (a["mutation_id"], b["mutation_id"]),
                )
                evaluated.append((row, runs))
        qualified = [
            (row, runs) for row, runs in evaluated if row["decision"] == "ACCEPTED"
        ]

        def quality(pair):
            row = pair[0]
            current_gate = row["gate"]["opponents"][0]
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
                "candidate_accuracies": [r["accuracy"] for r in candidate_rows],
                "selected_accuracy": selected_accuracy,
                "selected_customer": selected_source,
                "candidates": candidate_rows,
                "incumbent_episodes": [r.to_dict() for r in incumbent],
            },
            "service_phase": {
                "carrier": "skill_memory_v2",
                "old_accuracy": selected_accuracy,
                "final_accuracy": _accuracy(final_runs),
                "proposed_accuracy": proposed_accuracy,
                "accepted": winner is not None,
                "acceptance_mode": "e_only_mechanism_smoke"
                if smoke
                else "statistical_validation_gated",
                "validation_evaluated": any(r["gate"] for r in board) and not smoke,
                "selection": {
                    "reason": "qualified candidate promoted"
                    if winner
                    else "no candidate passed all screen/promotion/replay gates"
                },
                "skill_count_before": len(before_s.skills),
                "skill_count_after": len(service.skills),
                "candidates": board,
                "diagnosis": diagnosis,
                "opponents_replayed": [label for label, _, _ in opponents],
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
            "failure_matrix": [
                r for row in board for r in row.get("failure_matrix", [])
            ],
            "history_summary": summarize_effects(effects),
            "statistical_policy": policy["statistical_gate"],
            "timing": {
                "validation_wall_clock_seconds": sum(
                    elapsed
                    for key, elapsed in timing["stages"].items()
                    if key.startswith(f"g{g:04d}-service_full_gate") and not smoke
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
):
    """Freeze and validate the final adaptive challenge using E-only evidence before H loads."""
    from .alternating import _context_episodes, _write_json_once

    journal = EvolutionJournal(output_directory, manifest_sha256)
    context = {
        "task_interactions": _context_episodes(
            result.final_evolution_episodes, runner, tasks
        ),
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
    if not all(
        validation.get(key) is True
        for key in (
            "preserves_facts",
            "preserves_objective",
            "interaction_only",
            "no_benchmark_leakage",
        )
    ):
        raise ValueError("fresh Customer semantic validation failed before H loading")
    customer = PromptStrategy(proposal["strategy"])
    _write_json_once(
        Path(output_directory) / "fresh-customer-proposal.json",
        {
            "schema_version": 3,
            "manifest_sha256": manifest_sha256,
            "evolver_input_sha256": sha256_json(context),
            "strategy": customer.text,
            "strategy_id": customer_strategy_id(customer),
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
    for index in range(4):
        cell = deepcopy(per_seed[0]["cells"][index])
        episodes = [
            e for result in per_seed for e in result["cells"][index]["episodes"]
        ]
        records = [EpisodeRecord.from_dict(e) for e in episodes]
        cell.update(
            accuracy=episode_metrics(records)["accuracy"],
            episodes=episodes,
            pass_power_k={str(k): pass_power_k(records, k) for k in (1, 2)},
        )
        cells.append(cell)
    summary["cells"] = cells
    if output_path:
        _write_json_atomic(Path(output_path), summary)
    return summary
