"""Prepare or explicitly execute Evolver-only old/new replay; never load τ tasks or rollouts."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from evotau.alternating import LLMAlternatingEvolvers, _write_json_once
from evotau.budget import RequestBudget
from evotau.evolution_artifacts import EvolutionJournal
from evotau.evolution_candidates import (
    DIRECT_MUTATOR_PROMPT,
    FAILURE_ANALYST_PROMPT,
    MECHANISM_DEDUP_PROMPT,
    MUTATOR_PROMPT,
    V2Providers,
)
from evotau.failure_analysis import (
    ANALYST_VERSION,
    analysis_outcome,
    diversity_outcome,
    select_distinct_hypotheses,
)
from evotau.skill_evolution import _candidate_outcome
from evotau.tau_provenance import sha256_json


def replay(bundle, output, execute=False):
    if bundle["context_sha256"] != sha256_json(bundle["context"]):
        raise ValueError("Replay evidence digest mismatch")
    ctx = bundle["context"]
    ids = [r["task"]["task_id"] for r in ctx["task_interactions"]]
    if set(ids) != set(bundle["e_task_ids"]) or ctx["generation"] != 0:
        raise ValueError("Replay must be frozen Gen0 E evidence")
    if any(
        r["seed"] != 1 or set(r["task"]) != {"task_id"}
        for r in ctx["task_interactions"]
    ):
        raise ValueError("Unexpected seed or hidden task metadata in replay")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "bundle_sha256": sha256_json(bundle),
        "model": bundle["model"],
        "request_args": bundle["request_args"],
        "candidate_budget": 3,
        "native_episodes": 0,
        "heldout_loaded": False,
        "algorithm_version": ANALYST_VERSION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompt_sha256": {
            name: sha256_json(prompt)
            for name, prompt in (
                ("old", DIRECT_MUTATOR_PROMPT),
                ("analyst", FAILURE_ANALYST_PROMPT),
                ("new", MUTATOR_PROMPT),
                ("diversity", MECHANISM_DEDUP_PROMPT),
            )
        },
        "calls_planned": "old:3; new:1 Analyst + 0/1 diversity + 0..3 Mutators; no validators/gates/rollouts",
    }
    _write_json_once(output / "replay-plan.json", plan)
    old_inputs = []
    base = deepcopy(ctx)
    for k in ("proposal_bias", "proposal_index", "archive_parent"):
        base.pop(k, None)
    for i, bias in enumerate(
        ("narrow_applicability", "minimal_behavior", "structural_decomposition")
    ):
        old_inputs.append(
            {**base, "proposal_index": i, "proposal_bias": bias, "archive_parent": None}
        )
    new_input = {
        **base,
        "algorithm_version": ANALYST_VERSION,
        "requested_hypotheses": 3,
    }
    _write_json_once(
        output / "prepared-inputs.json", {"old": old_inputs, "analyst": new_input}
    )
    if not execute:
        return {"real_requests_started": False, **plan}
    budget = RequestBudget(None)
    budget.enable_live_usage(output / "api-usage-live.json")
    args = {k: v for k, v in bundle["request_args"].items() if k != "tool_count"}
    provider = V2Providers(
        LLMAlternatingEvolvers(
            model=bundle["model"],
            model_args=args,
            request_budget=budget,
            output_directory=output,
        )
    )
    journal = EvolutionJournal(output, sha256_json(plan))
    results = {"old": [], "new": []}
    for i, inp in enumerate(old_inputs):
        results["old"].append(
            journal.freeze(
                f"old-{i}",
                inp,
                lambda inp=inp: _candidate_outcome(
                    lambda: provider.propose_skill_mutation(inp), inp
                ),
            )
        )
    analysis = journal.freeze(
        "analysis",
        new_input,
        lambda: analysis_outcome(
            lambda: provider.analyze_service_failures(new_input), new_input
        ),
    )
    results["analysis"] = analysis
    hs = analysis["hypotheses"]
    diversity_ctx = {"algorithm_version": ANALYST_VERSION, "hypotheses": hs}
    reviewed = (
        journal.freeze(
            "diversity",
            diversity_ctx,
            lambda: diversity_outcome(
                lambda: provider.deduplicate_failure_hypotheses(diversity_ctx), hs
            ),
        )
        if len(hs) > 1
        else {"status": "VALID", "comparisons": []}
    )
    selection = (
        select_distinct_hypotheses(hs, {"comparisons": reviewed["comparisons"]}, 3)
        if reviewed["status"] == "VALID"
        else {"assigned_hypotheses": []}
    )
    results["selection"] = {**selection, "review": reviewed}
    for i, h in enumerate(selection["assigned_hypotheses"]):
        inp = {
            **base,
            "algorithm_version": ANALYST_VERSION,
            "proposal_index": i,
            "assigned_hypothesis": h,
        }
        results["new"].append(
            journal.freeze(
                f"new-{i}",
                inp,
                lambda inp=inp: _candidate_outcome(
                    lambda: provider.propose_skill_mutation(inp), inp
                ),
            )
        )
    _write_json_once(output / "replay-results.json", results)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--execute",
        action="store_true",
        help="Paid Evolver-only requests; explicit future authorization required",
    )
    args = p.parse_args()
    print(
        json.dumps(
            replay(json.loads(args.source.read_text()), args.output, args.execute),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
