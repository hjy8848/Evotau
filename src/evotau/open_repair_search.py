"""Independent open evidence-guided generation; no direction taxonomy or UCB."""

import json
from copy import deepcopy
from math import isfinite

from .customer_diagnostic import RepairConditionedTrials
from .customer_skills import proxy_tokens
from .evolution_archive import BanditTrialLedger
from .repair_feedback import validate_pair
from .tau_provenance import sha256_json

PROTOCOL = "open_repair_conditioned_search_v1"
DEFAULT = {
    "protocol_version": PROTOCOL,
    "mode": "open_rc",
    "max_trials_per_generation": 2,
    "feedback_seeds": [1, 2],
    "min_replications": 2,
    "max_context_tokens": 30000,
}


def validate_policy(raw):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULT):
        raise ValueError("unknown open repair search policy")
    p = {**deepcopy(DEFAULT), **deepcopy(raw)}
    if p["protocol_version"] != PROTOCOL or p["mode"] not in ("open_rc", "free_search"):
        raise ValueError("unknown open repair search identity")
    for k in ("max_trials_per_generation", "min_replications", "max_context_tokens"):
        if type(p[k]) is not int or p[k] < 1:
            raise ValueError("invalid open search budget")
    seeds = p["feedback_seeds"]
    if (
        not isinstance(seeds, list)
        or any(type(s) is not int or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
        or not 2 <= p["min_replications"] <= len(seeds)
    ):
        raise ValueError("open feedback requires replicated predeclared seeds")
    return p


class OpenRepairTrials(RepairConditionedTrials):
    """Reuse strict rollout/validity/reviewer feedback, never call choose_arm."""

    def __init__(
        self,
        *,
        policy,
        customer_policy,
        tasks,
        runner,
        providers,
        domain_policy,
        root,
        manifest_sha,
        budget=None,
    ):
        self.policy = validate_policy(policy)
        self.customer_policy, self.tasks = customer_policy, tasks
        self.runner, self.providers, self.domain_policy = (
            runner,
            providers,
            domain_policy,
        )
        self.root, self.budget = root, budget
        self.state = {
            "protocol_version": PROTOCOL,
            "policy_sha256": sha256_json(self.policy),
            "trials": [],
        }
        self.ledger = BanditTrialLedger(
            root, manifest_sha, tasks, protocol=PROTOCOL, directory="open-repair-trials"
        )
        self.active = None
        self.discovery_handoff = True
        self.include_candidate_program = True

    @property
    def protocol(self):
        return PROTOCOL

    def restore_state(self, value):
        if (
            set(value) != set(self.state)
            or value["protocol_version"] != PROTOCOL
            or value["policy_sha256"] != sha256_json(self.policy)
        ):
            raise ValueError("open search checkpoint identity mismatch")
        old = self.state
        self.state = {**old, "trials": []}
        try:
            for t in value["trials"]:
                if any(x["trial_id"] == t["trial_id"] for x in self.state["trials"]):
                    raise ValueError("duplicate open trial")
                self.finalize_trial(t)
            return deepcopy(self.state)
        finally:
            self.state = old

    def finalize_trial(self, trial):
        if (
            trial["protocol_version"] != PROTOCOL
            or trial["arm"] is not None
            or trial["reward"] != trial["feedback"]["reward"]
        ):
            raise ValueError("invalid open trial identity")
        for value in trial["cost"].values():
            if value is not None and (
                type(value) not in (int, float) or not isfinite(value) or value < 0
            ):
                raise ValueError("invalid actual open trial cost")
        for old in self.state["trials"]:
            if old["trial_id"] == trial["trial_id"]:
                if old != trial:
                    raise ValueError("conflicting open trial")
                return
        if trial["reward"]:
            f = trial["feedback"]
            if (
                trial["candidate_validity"] != "valid"
                or f["status"] != "discovery"
                or f["service_pair_id"] != trial["service_pair_id"]
                or not f.get("reviewer_provenance")
                or not f.get("review")
                or not f["discovery_keys"]
            ):
                raise ValueError("unsupported open discovery reward")
            from .repair_feedback import discovery_feedback

            verified = discovery_feedback(
                trial["observations"],
                trial["review_context"],
                f["review"],
                reviewer_provenance=f["reviewer_provenance"],
                prior_keys=[],
                min_replications=self.policy["min_replications"],
            )
            if not set(f["discovery_keys"]) <= set(verified["discovery_keys"]):
                raise ValueError("unverified restored open discovery")
            previous = {
                k
                for t in self.state["trials"]
                for k in t["feedback"].get("discovery_keys", [])
            }
            if previous.intersection(f["discovery_keys"]):
                raise ValueError("duplicate discovery credit")
        self.state["trials"].append(deepcopy(trial))

    def begin(self, g, i, context, pair, stage):
        if i >= self.policy["max_trials_per_generation"]:
            raise ValueError("open search candidate budget exceeded")
        if pair:
            validate_pair(pair)
            if (
                pair["created_generation"] >= g
                or pair["kind"] != "promoted"
                or any(
                    c["task_id"] not in self.tasks
                    for rows in pair["source_cells"].values()
                    for c in rows
                )
            ):
                raise ValueError(
                    "open search requires a previous promoted E-only repair"
                )
        ident = f"g{g:04d}-c{i:04d}"
        choice = stage(
            f"rc-{i}-choice",
            {"state": self.state, "pair": pair, "config": self.policy},
            lambda: {
                "arm": None,
                "service_pair_id": pair["pair_id"] if pair else None,
                "frontier_status": "repair_pair" if pair else "no_repair_pair",
                "search_mode": self.policy["mode"],
            },
        )
        start = stage(
            f"rc-{i}-start",
            {"choice": choice, "trial_id": ident},
            lambda: {
                "cost_before": self.costs(),
                "choice": choice,
                "api_usage_before": self.budget.api_usage_by_call_name()
                if self.budget
                else None,
            },
        )
        self.ledger.publish(ident, "start", start)
        self.active = {
            "trial_id": ident,
            "generation": g,
            "choice": choice,
            "cost_before": start["cost_before"],
            "api_usage_before": start["api_usage_before"],
        }
        search = {
            "protocol_version": PROTOCOL,
            "mode": self.policy["mode"],
            "frontier_status": "ordinary_open_exploration",
            "candidate_index": i,
            "max_context_tokens": self.policy["max_context_tokens"],
            "past_attempts": [
                {
                    "procedure_id": t["procedure_id"],
                    "proposed_program": t.get("customer_proposal"),
                    "validity": t["candidate_validity"],
                    "rejection": t["rejection"],
                    "discovery_status": t["feedback"]["status"],
                    "mechanisms": [
                        d["mechanism"]
                        for d in t["feedback"].get("review", {}).get("discoveries", [])
                    ],
                }
                for t in self.state["trials"][-20:]
            ],
        }
        if self.policy["mode"] == "free_search":
            search["past_attempts"] = [
                {
                    k: t[k]
                    for k in (
                        "procedure_id",
                        "proposed_program",
                        "validity",
                        "rejection",
                    )
                }
                for t in search["past_attempts"]
            ]
        if pair and self.policy["mode"] == "open_rc":
            search.update(
                frontier_status="repair_pair",
                repair_pair=pair,
                repair_evidence=getattr(self, "repair_evidence", None),
            )
        if proxy_tokens(json.dumps(search)) > self.policy["max_context_tokens"]:
            raise ValueError("open repair evidence exceeds frozen allowance")
        result = {**context, "open_repair_search": search}
        if proxy_tokens(json.dumps(result)) > self.policy["max_context_tokens"]:
            raise ValueError("combined open repair context exceeds frozen allowance")
        return result
