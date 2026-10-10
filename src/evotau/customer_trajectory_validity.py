"""Auditable post-rollout Customer validity, separate from native reward."""

from .tau_provenance import sha256_json

TRAJECTORY_VALIDATOR_PROMPT = """Review actual Customer behavior, not Service effectiveness.
Compare each complete E trajectory with its unchanged original user_scenario and public policy.
Do not use gold actions or guess omitted goals. Distinguish Service mistakes from Customer
violations. valid: Customer preserved fixed goals, identity, budget, facts, conditional consent,
and promptly supplied necessary requested information; reasonable lawful pressure is allowed.
invalid: actual Customer changed fixed goals/facts/consent, invented requests or refused necessary
cooperation, stalled or stopped just to cause failure. uncertain: evidence cannot establish validity.
Never call Service failure itself Customer invalidity. Return exactly {"cells":[{"task_id":"...",
"seed":1,"status":"valid|invalid|uncertain","reason":"...","evidence_message_indices":[1]}]}.
Return exactly one cell per supplied task×seed; cite supplied USER message indices supporting
your conclusion (at least one). These are observations, not proven causal Skill effects."""


def trajectory_context(rows, skill, policy):
    cells = []
    for row in rows:
        cells.append(
            {
                "task_id": row["task"]["task_id"],
                "seed": row["seed"],
                "user_scenario": row["task"]["user_scenario"],
                "trajectory_ref": row["trajectory_ref"],
                "trajectory": row["trajectory"],
            }
        )
    return {
        "customer_protocol": "task_faithful_customer_skill_v1",
        "skill": skill,
        "service_policy": policy,
        "cells": cells,
    }


def validate_trajectory_report(report, context):
    if (
        not isinstance(report, dict)
        or set(report) != {"cells"}
        or not isinstance(report["cells"], list)
    ):
        raise ValueError("invalid Customer trajectory report")
    sources = {(c["task_id"], c["seed"]): c for c in context["cells"]}
    if len(sources) != len(context["cells"]):
        raise ValueError("duplicate source trajectory cell")
    seen = set()
    for cell in report["cells"]:
        if not isinstance(cell, dict) or set(cell) != {
            "task_id",
            "seed",
            "status",
            "reason",
            "evidence_message_indices",
        }:
            raise ValueError("invalid Customer trajectory cell fields")
        if not isinstance(cell["task_id"], str) or type(cell["seed"]) is not int:
            raise ValueError("invalid Customer trajectory cell identity")
        key = cell["task_id"], cell["seed"]
        if (
            key not in sources
            or key in seen
            or cell["status"] not in ("valid", "invalid", "uncertain")
        ):
            raise ValueError("unknown/duplicate trajectory cell or status")
        seen.add(key)
        if not isinstance(cell["reason"], str) or not cell["reason"].strip():
            raise ValueError("trajectory review requires a reason")
        indices = cell["evidence_message_indices"]
        messages = sources[key]["trajectory"]["messages"]
        if (
            not isinstance(indices, list)
            or not indices
            or any(
                type(i) is not int
                or not 0 <= i < len(messages)
                or messages[i].get("role") != "user"
                for i in indices
            )
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("trajectory evidence must cite original Customer messages")
    if seen != set(sources):
        raise ValueError("Customer review omitted evaluation cells")
    return report


def assess_trajectories(context, callback):
    """Missing evidence is uncertain; malformed/provider errors remain exceptions."""
    usable, unavailable = [], []
    for cell in context["cells"]:
        if not cell["user_scenario"].strip() or not any(
            m.get("role") == "user" for m in cell["trajectory"]["messages"]
        ):
            unavailable.append(
                {
                    "task_id": cell["task_id"],
                    "seed": cell["seed"],
                    "status": "uncertain",
                    "reason": "missing original scenario or Customer trajectory",
                    "evidence_message_indices": [],
                }
            )
        else:
            usable.append(cell)
    report = {"cells": []}
    if usable:
        subcontext = {**context, "cells": usable}
        report = validate_trajectory_report(callback(subcontext), subcontext)
    cells = report["cells"] + unavailable
    status = (
        "invalid"
        if any(c["status"] == "invalid" for c in cells)
        else "uncertain"
        if any(c["status"] == "uncertain" for c in cells)
        else "valid"
    )
    return {
        "status": status,
        "cells": cells,
        "input_sha256": sha256_json(context),
        "semantic_validity_is_model_assessed": True,
    }


def paired_customer_feedback(before, after, *, validity):
    from .records import EpisodeStatus

    left = {(r.task_id, r.seed): r for r in before}
    right = {(r.task_id, r.seed): r for r in after}
    if (
        not left
        or len(left) != len(before)
        or len(right) != len(after)
        or left.keys() != right.keys()
    ):
        raise ValueError("Customer comparison requires complete unique task×seed pairs")
    if any(left[k].service_strategy_id != right[k].service_strategy_id for k in left):
        raise ValueError("Customer comparison requires identical Service")
    if any(
        type(r.task_success) is not bool or r.status != EpisodeStatus.COMPLETE
        for r in (*before, *after)
    ):
        raise ValueError("Customer comparison requires known native outcomes")
    reviewed = {(c["task_id"], c["seed"]): c for c in validity["cells"]}
    if reviewed.keys() != right.keys() or len(reviewed) != len(validity["cells"]):
        raise ValueError("Customer validity must cover all paired evaluation cells")
    if any(
        c["status"] not in ("valid", "invalid", "uncertain") for c in reviewed.values()
    ):
        raise ValueError("unknown Customer validity status")
    status = (
        "invalid"
        if any(c["status"] == "invalid" for c in reviewed.values())
        else "uncertain"
        if any(c["status"] == "uncertain" for c in reviewed.values())
        else "valid"
    )
    if validity["status"] != status:
        raise ValueError("Customer validity aggregate disagrees with cell verdicts")
    new = [
        {"task_id": k[0], "seed": k[1]}
        for k in left
        if left[k].task_success and not right[k].task_success
    ]
    recovered = [
        {"task_id": k[0], "seed": k[1]}
        for k in left
        if not left[k].task_success and right[k].task_success
    ]
    delta = (len(recovered) - len(new)) / len(left)
    return {
        "candidate_validity": validity["status"],
        "new_failures": new,
        "recovered_cells": recovered,
        "paired_accuracy_delta": delta,
        "invalid_customer_episodes": [
            c for c in validity["cells"] if c["status"] == "invalid"
        ],
        "uncertain_customer_episodes": [
            c for c in validity["cells"] if c["status"] == "uncertain"
        ],
        "eligible_for_selection": validity["status"] == "valid",
        "causal_service_failure_confirmed": False,
    }
