"""E-only Customer search with a frozen native Service; no Service or H search."""

from .customer_evolution import (
    evaluate_candidate,
    generate_candidates,
    prepare_candidate,
)
from .customer_skills import PROTOCOL
from .evolution_archive import CustomerAttemptHistory
from .evolution_artifacts import EvolutionJournal
from .strategies import PromptStrategy
from .tau_provenance import sha256_json


def run_customer_diagnostic(
    *,
    tasks,
    task_ids,
    runner,
    providers,
    service,
    policy,
    generations,
    count,
    seed,
    concurrency,
    domain_policy,
    output,
    manifest_sha,
    initial_customer=None,
    checkpoint_path=None,
):
    import json
    from pathlib import Path

    from .alternating import (
        _accuracy,
        _context_episodes,
        _run_panel,
        _write_json_atomic,
    )
    from .evolution_context import build_customer_evidence
    from .records import EpisodeRecord

    if set(tasks) != set(task_ids):
        raise ValueError("Customer diagnostic may load only the frozen E panel")
    providers.configure_customer({"customer_evolution": policy})
    journal = EvolutionJournal(output, manifest_sha)
    history = CustomerAttemptHistory(max_history=policy["max_history"])
    customer = initial_customer or PromptStrategy("")
    journal.freeze(
        "customer-diagnostic-plan",
        {
            "task_ids": task_ids,
            "seed": seed,
            "policy": policy,
            "generations": generations,
            "count": count,
            "concurrency": concurrency,
            "initial_customer": customer.to_dict(),
            "frozen_service": service.to_dict(),
            "domain_policy": domain_policy,
        },
        lambda: {"customer_protocol": PROTOCOL},
    )
    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path
        else Path(output) / "customer-diagnostic-checkpoint.json"
    )
    if checkpoint.is_symlink():
        raise ValueError("unsafe Customer diagnostic checkpoint")
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    if saved is not None and (
        saved.get("manifest_sha256") != manifest_sha
        or saved.get("customer_protocol") != PROTOCOL
        or saved.get("sha256")
        != sha256_json({k: v for k, v in saved.items() if k != "sha256"})
    ):
        raise ValueError("Customer diagnostic checkpoint identity/digest mismatch")
    if (
        saved is not None
        and not (
            journal.root
            / f"customer-diagnostic-g{saved['completed_generation']}-complete.json"
        ).exists()
    ):
        raise ValueError("Customer diagnostic checkpoint refers to missing generation")
    documents = []

    def freeze(name, inputs, callback):
        _write_json_atomic(
            Path(output) / "customer-diagnostic-stage.json",
            {
                "stage": name,
                "manifest_sha256": manifest_sha,
                "customer_protocol": PROTOCOL,
            },
        )
        return journal.freeze(name, inputs, callback)

    def panel(g, label, c):
        inputs = {
            "customer": c.to_dict(),
            "service": service.to_dict(),
            "ids": task_ids,
            "seed": seed,
        }
        result = freeze(
            f"customer-diagnostic-g{g}-{label}",
            inputs,
            lambda: {
                "episodes": [
                    r.to_dict()
                    for r in _run_panel(
                        runner,
                        tasks=tasks,
                        task_ids=task_ids,
                        seed=seed,
                        customer=c,
                        service=service,
                        panel_name=f"customer-diagnostic-g{g}-{label}",
                        max_parallel_episodes=concurrency,
                    )
                ]
            },
        )
        return tuple(EpisodeRecord.from_dict(v) for v in result["episodes"])

    for g in range(generations):
        incumbent = panel(g, "incumbent", customer)
        context = {
            "customer_evolution": policy,
            "generation": g,
            "task_interactions": build_customer_evidence(
                _context_episodes(incumbent, runner, tasks),
                representative_cases=policy["representative_cases"],
                case_chars=policy["case_chars"],
                customer_protocol=PROTOCOL,
            ),
            "current_customer_strategy": customer.text,
            "current_service": service.to_dict(),
            "customer_attempt_history": history.entries,
            "service_policy": domain_policy,
        }
        proposed = freeze(
            f"customer-diagnostic-g{g}-proposals",
            context,
            lambda context=context: generate_candidates(providers, context, count),
        )
        selected_accuracy = _accuracy(incumbent)
        rows = []
        if proposed.get("generation_rejection"):
            rows.append(proposed["generation_rejection"])
        tested_procedures = {entry.get("procedure_id") for entry in history.entries}
        for i, proposal in enumerate(proposed["candidates"]):
            row, skill = prepare_candidate(proposal, context, policy)
            if skill is not None and (
                skill.compile().text == customer.text
                or skill.procedure_id in tested_procedures
            ):
                row.update(
                    candidate_validity="invalid",
                    pre_rollout_rejection="duplicate Customer procedure",
                )
                skill = None
            if skill is not None:
                tested_procedures.add(skill.procedure_id)
            row, _ = evaluate_candidate(
                row,
                skill,
                context=context,
                incumbent=incumbent,
                current_service=service,
                tasks=tasks,
                runner=runner,
                providers=providers,
                domain_policy=domain_policy,
                panel=lambda c, s, i=i, g=g: panel(g, f"candidate-{i}", c),
                stage=lambda n, inputs, cb, g=g, i=i: freeze(
                    f"customer-diagnostic-g{g}-c{i}-{n}", inputs, cb
                ),
            )
            if row["eligible_for_selection"] and row["accuracy"] < selected_accuracy:
                customer, selected_accuracy = skill.compile(), row["accuracy"]
            rows.append(row)
        for row in rows:
            row["selected"] = bool(
                row.get("eligible_for_selection")
                and row.get("strategy", {}).get("text") == customer.text
            )
            if row.get("eligible_for_selection"):
                row["selection_reason"] = (
                    "selected: strict native E accuracy decrease"
                    if row["selected"]
                    else "no strict native E accuracy decrease"
                    if row["accuracy"] >= _accuracy(incumbent)
                    else "another eligible candidate has lower native E accuracy"
                )
            history.add(row, g)
        doc = freeze(
            f"customer-diagnostic-g{g}-complete",
            {
                "rows": rows,
                "customer": customer.to_dict(),
                "service": service.to_dict(),
            },
            lambda rows=rows, selected_accuracy=selected_accuracy, g=g, incumbent=incumbent: {
                "generation": g,
                "incumbent_accuracy": _accuracy(incumbent),
                "selected_accuracy": selected_accuracy,
                "candidates": rows,
            },
        )
        documents.append(doc)
        state = {
            "customer_protocol": PROTOCOL,
            "manifest_sha256": manifest_sha,
            "completed_generation": g,
            "customer": customer.to_dict(),
            "frozen_service": service.to_dict(),
            "attempt_history": history.entries,
            "generation_sha256": sha256_json(doc),
        }
        state["sha256"] = sha256_json(state)
        if saved is not None and saved["completed_generation"] == g and saved != state:
            raise ValueError(
                "Customer diagnostic checkpoint differs from committed generation"
            )
        if saved is None or g > saved["completed_generation"]:
            _write_json_atomic(checkpoint, state)
    return {
        "status": "complete",
        "customer_protocol": PROTOCOL,
        "generations": documents,
        "customer": customer.to_dict(),
        "frozen_service": service.to_dict(),
        "attempt_history": history.entries,
        "validation_loaded": False,
        "heldout_loaded": False,
        "causal_service_failure_confirmed": False,
    }


class RepairConditionedTrials:
    """Opt-in E-only dual-Service trials, also used by the alternating loop.

    Does not change run_customer_diagnostic's frozen single-Service contract.
    The caller's immutable stage/panel callbacks own resume and provider accounting.
    """

    def __init__(self, *, policy, customer_policy, tasks, runner, providers, domain_policy,
                 root, manifest_sha, budget=None):
        from .evolution_archive import BanditTrialLedger
        from .repair_conditioned_bandit import new_state, validate_policy
        self.policy = validate_policy(policy)
        self.customer_policy, self.tasks = customer_policy, tasks
        self.runner, self.providers, self.domain_policy = runner, providers, domain_policy
        self.root, self.budget = root, budget
        self.state = new_state(self.policy)
        self.ledger = BanditTrialLedger(root, manifest_sha, tasks)
        self.active = None

    def costs(self):
        """Cumulative actual run accounting, never a request-count estimate.

        Unknown token usage stays unknown. In Mock runs the extra counters are
        explicitly simulated; those are not falsely reported as paid API calls.
        """
        from pathlib import Path
        root = Path(self.root)
        simulated = hasattr(self.runner, "calls") and hasattr(self.runner, "cache")
        counts = {"native_episodes": len(self.runner.calls) if simulated else
                  len(list(root.glob("episodes/*/episode-record.json"))),
                  "native_attempts": len(self.runner.calls) if simulated else
                  len(list(root.glob("episodes/*"))),
                  "simulated_provider_calls": len(self.providers.calls) if simulated and hasattr(self.providers, "calls") else None,
                  "api_calls": 0 if simulated else None, "prompt_tokens": None,
                  "completion_tokens": None, "denied_calls": None, "in_flight": None,
                  "review_calls": None}
        if self.budget is not None:
            snap = self.budget.snapshot()
            counts.update(api_calls=snap.attempts, prompt_tokens=snap.prompt_tokens,
                          completion_tokens=snap.completion_tokens, denied_calls=snap.denied,
                          in_flight=snap.in_flight, usage_unavailable=snap.usage_unavailable)
            counts["review_calls"] = sum(c["calls"] for name, c in self.budget.api_usage_by_call_name().items()
                                         if "validator" in name or "repair_discovery_review" in name)
        return counts

    def begin(self, g, i, context, pair, stage):
        from .customer_skills import proxy_tokens
        from .repair_conditioned_bandit import choose_arm
        from .repair_feedback import validate_pair
        if i >= self.policy["max_trials_per_generation"]:
            raise ValueError("RC per-generation trial budget exceeded")
        if pair:
            validate_pair(pair)
            if any(c["task_id"] not in self.tasks for records in pair["source_cells"].values() for c in records):
                raise ValueError("repair frontier contains non-E cells")
            if pair["created_generation"] >= g or pair["kind"] != "promoted":
                raise ValueError("Customer cannot use an unpromoted/current-generation Service repair")
        ident = f"g{g:04d}-c{i:04d}"
        choice = stage(f"rc-{i}-choice", {"state": self.state, "pair": pair, "config": self.policy},
                       lambda: choose_arm(self.state, pair["pair_id"] if pair else None, self.policy))
        start = stage(f"rc-{i}-start", {"choice": choice, "trial_id": ident},
                      lambda: {"cost_before": self.costs(), "choice": choice,
                          "api_usage_before": self.budget.api_usage_by_call_name() if self.budget else None})
        self.ledger.publish(ident, "start", start)
        self.active = {"trial_id": ident, "generation": g, "choice": choice, "cost_before": start["cost_before"], "api_usage_before": start["api_usage_before"]}
        repair_context = {"frontier_status": choice["frontier_status"],
                          "service_pair_id": choice["service_pair_id"],
                          "previous_E_discoveries": [{"keys": t["feedback"].get("discovery_keys", []),
                              "status": t["feedback"].get("status"), "arm": t["arm"]}
                              for t in self.state["trials"][-20:] if t["service_pair_id"] == choice["service_pair_id"]]}
        if pair:
            repair_context["changed_service"] = {"before": pair["before"], "after": pair["after"]}
            repair_context["created_generation"] = pair["created_generation"]
            repair_context["previous_E_repair_results"] = pair["source_cells"]
        if proxy_tokens(__import__("json").dumps(repair_context)) > self.policy["max_context_tokens"]:
            raise ValueError("RC context exceeds frozen allowance; no silent evidence truncation")
        return {**context, "operator_family_hint": choice["arm"], "repair_context": repair_context}

    def _cost_delta(self):
        after, before = self.costs(), self.active["cost_before"]
        result = {}
        for k, value in after.items():
            old = before.get(k)
            result[k] = None if value is None or old is None else value - old
            if result[k] is not None and result[k] < 0:
                raise ValueError("RC actual cost counters reset across resume")
        return result

    def failure(self, error):
        if self.active:
            self.ledger.publish(self.active["trial_id"], "failure", {
                "error_type": type(error).__name__, "error": str(error),
                "cost": self._cost_delta(), "status": "unresolved_fail_closed",
                "retry_authorized": False, "choice": self.active["choice"]})

    def finish(self, row, skill, *, pair, stage, panel):
        from .alternating import _service_context_episodes
        from .customer_evolution import review_customer_runs
        from .customer_skills import proxy_tokens
        from .evolution_candidates import EvolverSchemaError
        from .evolution_context import build_repair_pair_evidence
        from .records import customer_strategy_id
        from .repair_conditioned_bandit import PROTOCOL, record_trial
        from .repair_feedback import (
            REPAIR_REVIEW_PROMPT,
            discovery_feedback,
            paired_observations,
            review_context,
        )
        from .service_skills import ServiceSkillMemoryV2

        feedback = {"status": "no_repair_pair" if pair is None else "invalid_or_uncertain_candidate",
                    "reward": 0, "discovery_keys": [], "causal_service_failure_confirmed": False}
        observations = None
        if pair and skill and row["eligible_for_selection"]:
            endpoints, validity, evidence = {}, {}, {}
            for name in ("before", "after"):
                service = ServiceSkillMemoryV2.from_mapping(pair[name])
                endpoints[name] = panel(name, self.policy["feedback_seeds"], skill.compile(), service)
                validity[name] = review_customer_runs(endpoints[name], skill=skill.to_dict(), tasks=self.tasks,
                    runner=self.runner, providers=self.providers, domain_policy=self.domain_policy,
                    policy=self.customer_policy,
                    stage=lambda label, inputs, cb, name=name: stage(name + "-" + label, inputs, cb))
                evidence[name] = _service_context_episodes(endpoints[name], self.runner, self.tasks)
            observations = paired_observations(endpoints["before"], endpoints["after"], pair=pair,
                customer_id=customer_strategy_id(skill.compile()), task_ids=list(self.tasks),
                seeds=self.policy["feedback_seeds"], validity=validity)
            evidence = build_repair_pair_evidence(evidence, observations,
                representative_tasks=self.customer_policy["representative_cases"],
                case_chars=self.customer_policy["case_chars"])
            prior = sorted({key for t in self.state["trials"] for key in t["feedback"].get("discovery_keys", [])})
            descriptions = [{"service_pair_id": t["service_pair_id"], "mechanism": d["mechanism"],
                "task_id": d["task_id"], "discovery_type": d["discovery_type"]}
                for t in self.state["trials"] for d in t["feedback"].get("review", {}).get("discoveries", [])]
            context = review_context(observations, pair, evidence, prior_discoveries=descriptions)
            def review():
                if (proxy_tokens(REPAIR_REVIEW_PROMPT + __import__("json").dumps(context)) > self.policy["max_context_tokens"]
                        or not hasattr(self.providers, "review_repair_discoveries")):
                    return {"report": None, "provenance": None, "reason": "review unavailable or full retained evidence exceeds allowance"}
                try:
                    report = self.providers.review_repair_discoveries(context)
                except EvolverSchemaError as error:
                    return {"report": None, "provenance": None, "schema_error": str(error),
                            "diagnostics_ref": getattr(error, "diagnostics_ref", None)}
                wrapped = getattr(self.providers, "provider", None)
                return {"report": report, "provenance": {
                    "model": getattr(wrapped, "model", "scripted/no real provider"),
                    "response_ref": str(getattr(self.providers, "last_call_directory", "scripted")),
                    "input_sha256": sha256_json(context), "response_sha256": sha256_json(report)}}
            reviewed = stage("repair-evidence-review", context, review)
            feedback = discovery_feedback(observations, context, reviewed["report"],
                reviewer_provenance=reviewed["provenance"], prior_keys=prior,
                min_replications=self.policy["min_replications"])
            feedback["review_stage"] = reviewed
        trial = {"protocol_version": PROTOCOL, "trial_id": self.active["trial_id"],
                 "generation": self.active["generation"], "arm": self.active["choice"]["arm"],
                 "choice": self.active["choice"], "service_pair_id": pair["pair_id"] if pair else None,
                 "procedure_id": row.get("procedure_id"), "candidate_id": row.get("candidate_id"),
                 "candidate_validity": row["candidate_validity"], "rejection": row.get("pre_rollout_rejection"),
                 "candidate_evaluation_sha256": sha256_json(row),
                 "native_accuracy": row.get("accuracy"),
                 "source_E_cells": [{"task_id": r["task_id"], "seed": r["seed"],
                    "task_success": r["task_success"], "trajectory_ref": r["trajectory_ref"],
                    "record_sha256": sha256_json(r)} for r in row.get("episodes", [])],
                 "observations": observations, "feedback": feedback, "reward": feedback["reward"]}
        final = stage("trial-final", trial, lambda: {**trial, "cost": self._cost_delta(),
            "api_usage_before": self.active.get("api_usage_before"),
            "api_usage_after": self.budget.api_usage_by_call_name() if self.budget else None})
        self.ledger.publish(self.active["trial_id"], "final", final)
        self.state = record_trial(self.state, final)
        self.active = None
        return final
