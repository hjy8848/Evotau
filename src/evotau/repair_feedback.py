"""E-only observational Service repair differences; no causal or promotion credit."""

from .records import EpisodeStatus
from .tau_provenance import sha256_json

REPAIR_REVIEW_PROMPT = """Review only supplied E evidence of a PROMOTED Service repair pair.
Same task/seed/Customer procedure does NOT mean identical dialogues or causal proof.
Find repair-related residual weaknesses (both fail) or credible regressions (pass to fail),
only if the SAME observable mechanism is supported across the predeclared independent seeds.
Do not reward termination randomness, mere double failures, changed Customer facts, speculation,
or repeat a supplied prior discovery. Never infer hidden goals or gold answers.
Return exactly {"discoveries":[...]}. Empty is a legitimate result.
Each discovery has exactly mechanism (a concrete observable behavioral mechanism), task_id,
discovery_type (residual|regression), seeds (unique integers), evidence_refs, repair_related
(boolean), novel (boolean, compared with supplied prior mechanisms), reason (nonempty string). Each evidence_ref copies EXACTLY one supplied ref including
endpoint (before|after), task_id, seed, trajectory_ref, projected_message_index, message_sha256.
Cite observable Service/tool behavior at BOTH endpoints for EVERY replication seed.
repair_related must describe relevance to the supplied changed Service guidance, not just failure.
This review is a hypothesis assessment, not proof of a causal Service effect."""


def service_pair(
    before, after, *, generation, promoted, before_records=(), after_records=()
):
    from .records import service_strategy_id

    if not promoted or service_strategy_id(before) == service_strategy_id(after):
        return None
    value = {
        "before": before.to_dict(),
        "after": after.to_dict(),
        "before_id": service_strategy_id(before),
        "after_id": service_strategy_id(after),
        "created_generation": generation,
        "kind": "promoted",
        "evidence_panel": "E",
        "source_cells": {
            name: [
                {
                    "task_id": r.task_id,
                    "seed": r.seed,
                    "customer_strategy_id": r.customer_strategy_id,
                    "service_strategy_id": r.service_strategy_id,
                    "task_success": r.task_success,
                    "trajectory_ref": r.trajectory_ref,
                    "record_sha256": sha256_json(r.to_dict()),
                }
                for r in records
            ]
            for name, records in (("before", before_records), ("after", after_records))
        },
    }
    value["pair_id"] = sha256_json(value)
    return value


def validate_pair(pair):
    from .records import service_strategy_id
    from .service_skills import ServiceSkillMemoryV2

    if (
        not isinstance(pair, dict)
        or set(pair)
        != {
            "before",
            "after",
            "before_id",
            "after_id",
            "created_generation",
            "kind",
            "evidence_panel",
            "pair_id",
            "source_cells",
        }
        or pair["evidence_panel"] != "E"
        or pair["before_id"] == pair["after_id"]
        or pair["kind"] not in ("promoted", "shadow_repair")
        or pair["pair_id"]
        != sha256_json({k: v for k, v in pair.items() if k != "pair_id"})
        or any(
            service_strategy_id(ServiceSkillMemoryV2.from_mapping(pair[key]))
            != pair[key + "_id"]
            for key in ("before", "after")
        )
    ):
        raise ValueError("invalid frozen Service pair")
    return pair


def paired_observations(before, after, *, pair, customer_id, task_ids, seeds, validity):
    """Structural mismatches are errors. Non-valid observed Customers remain unscored evidence."""
    validate_pair(pair)
    expected = {(str(t), s) for t in task_ids for s in seeds}
    if (
        not expected
        or len(expected) != len(task_ids) * len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("repair panel requires unique E tasks and seeds")
    reports = {}
    endpoints = {}
    for name, records in (("before", before), ("after", after)):
        cells = {(r.task_id, r.seed): r for r in records}
        if len(cells) != len(records) or cells.keys() != expected:
            raise ValueError(
                "repair feedback requires complete frozen task×seed coverage"
            )
        if any(
            r.service_strategy_id != pair[name + "_id"]
            or r.customer_strategy_id != customer_id
            or r.status != EpisodeStatus.COMPLETE
            or type(r.task_success) is not bool
            for r in records
        ):
            raise ValueError("repair feedback episode identity/status mismatch")
        report = validity[name]
        reviewed = {(c["task_id"], c["seed"]): c for c in report["cells"]}
        if (
            len(reviewed) != len(report["cells"])
            or reviewed.keys() != expected
            or any(
                c["status"] not in ("valid", "invalid", "uncertain")
                for c in reviewed.values()
            )
        ):
            raise ValueError("repair validity coverage mismatch")
        aggregate = (
            "invalid"
            if any(c["status"] == "invalid" for c in reviewed.values())
            else "uncertain"
            if any(c["status"] == "uncertain" for c in reviewed.values())
            else "valid"
        )
        if aggregate != report["status"]:
            raise ValueError("repair validity aggregate mismatch")
        reports[name], endpoints[name] = reviewed, cells
    rows = []
    for t, s in sorted(expected):
        a, b = endpoints["before"][t, s], endpoints["after"][t, s]
        classification = {
            (False, True): "repaired",
            (False, False): "residual",
            (True, False): "regression",
            (True, True): "stable",
        }[a.task_success, b.task_success]
        rows.append(
            {
                "task_id": t,
                "seed": s,
                "classification": classification,
                "repair_related": None,
                "before": a.to_dict(),
                "after": b.to_dict(),
                "customer_validity": {name: reports[name][t, s] for name in reports},
            }
        )
    return {
        "panel": "E",
        "service_pair_id": pair["pair_id"],
        "pair_kind": pair["kind"],
        "customer_strategy_id": customer_id,
        "cells": rows,
        "causal_service_failure_confirmed": False,
    }


def review_context(observations, pair, rows, *, prior_discoveries):
    """Public, E-only projection. Callers supply existing Service-safe evidence rows."""
    expected = {(c["task_id"], c["seed"]) for c in observations["cells"]}
    evidence = []
    for endpoint in ("before", "after"):
        for row in rows[endpoint]:
            if (row["task"]["task_id"], row["seed"]) not in expected:
                raise ValueError("non-E evidence supplied to repair review")
            original = next(
                c[endpoint]
                for c in observations["cells"]
                if c["task_id"] == row["task"]["task_id"] and c["seed"] == row["seed"]
            )
            if (
                not row["trajectory_ref"]
                or row["trajectory_ref"] != original["trajectory_ref"]
            ):
                raise ValueError(
                    "repair evidence refers to a different episode trajectory"
                )
            messages = []
            for message in row["trajectory"]["messages"]:
                ref = {
                    "endpoint": endpoint,
                    "task_id": row["task"]["task_id"],
                    "seed": row["seed"],
                    "trajectory_ref": row["trajectory_ref"],
                    **message["evidence_ref"],
                }
                messages.append(
                    {k: v for k, v in message.items() if k != "evidence_ref"}
                    | {"evidence_ref": ref}
                )
            evidence.append(
                {
                    "endpoint": endpoint,
                    "task_id": row["task"]["task_id"],
                    "seed": row["seed"],
                    "messages": messages,
                }
            )
    return {
        "panel": "E",
        "pair": pair,
        "observations": {
            **observations,
            "cells": [
                {
                    **c,
                    **{
                        name: {
                            k: c[name][k]
                            for k in (
                                "episode_id",
                                "task_id",
                                "seed",
                                "customer_strategy_id",
                                "service_strategy_id",
                                "task_success",
                                "native_reward",
                                "trajectory_ref",
                                "termination_reason",
                            )
                        }
                        for name in ("before", "after")
                    },
                }
                for c in observations["cells"]
            ],
        },
        "evidence": evidence,
        "prior_discoveries": prior_discoveries,
    }


def validate_review(report, context):
    if (
        not isinstance(report, dict)
        or set(report) != {"discoveries"}
        or not isinstance(report["discoveries"], list)
    ):
        raise ValueError("invalid repair reviewer schema")
    registry = {
        sha256_json(m["evidence_ref"]): m
        for row in context["evidence"]
        for m in row["messages"]
    }
    cells = {(c["task_id"], c["seed"]): c for c in context["observations"]["cells"]}
    for item in report["discoveries"]:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "mechanism",
                "task_id",
                "discovery_type",
                "seeds",
                "evidence_refs",
                "repair_related",
                "novel",
                "reason",
            }
            or item["discovery_type"] not in ("residual", "regression")
            or any(
                not isinstance(item[k], str) or not item[k].strip()
                for k in ("mechanism", "task_id", "reason")
            )
            or type(item["repair_related"]) is not bool
            or type(item["novel"]) is not bool
            or not isinstance(item["seeds"], list)
            or not item["seeds"]
            or any(type(s) is not int for s in item["seeds"])
            or len(set(item["seeds"])) != len(item["seeds"])
            or any(
                (item["task_id"], s) not in cells
                or cells[item["task_id"], s]["classification"] != item["discovery_type"]
                for s in item["seeds"]
            )
            or not isinstance(item["evidence_refs"], list)
        ):
            raise ValueError("invalid repair discovery labels/coverage")
        cited = set()
        for ref in item["evidence_refs"]:
            message = registry.get(sha256_json(ref))
            if (
                message is None
                or ref["task_id"] != item["task_id"]
                or ref["seed"] not in item["seeds"]
            ):
                raise ValueError("unbound repair evidence reference")
            if message.get("role") in ("assistant", "tool"):
                cited.add((ref["endpoint"], ref["seed"]))
        if cited != {
            (endpoint, s) for endpoint in ("before", "after") for s in item["seeds"]
        }:
            raise ValueError(
                "repair discovery needs original Service/tool evidence at both endpoints"
            )
    return report


def discovery_feedback(
    observations, context, report, *, reviewer_provenance, prior_keys, min_replications
):
    """Conservative binary credit, independently replicated labels; never a gate replacement."""
    if type(min_replications) is not int or min_replications < 2:
        raise ValueError("discovery requires at least two replications")
    base = {
        "service_pair_id": observations["service_pair_id"],
        "status": "inconclusive",
        "reward": 0,
        "discovery_keys": [],
        "observations_sha256": sha256_json(observations),
        "causal_service_failure_confirmed": False,
        "reviewer_provenance": reviewer_provenance,
    }
    if observations["pair_kind"] != "promoted":
        return {**base, "status": "shadow_diagnostic"}
    if any(
        c["status"] != "valid"
        for row in observations["cells"]
        for c in row["customer_validity"].values()
    ):
        return {**base, "status": "invalid_or_uncertain_customer"}
    if report is None or not reviewer_provenance:
        return {
            **base,
            "status": "pending",
            "reason": "no repair-related reviewer support",
        }
    if (
        reviewer_provenance.get("input_sha256") != sha256_json(context)
        or reviewer_provenance.get("response_sha256") != sha256_json(report)
        or not reviewer_provenance.get("model")
        or not reviewer_provenance.get("response_ref")
    ):
        raise ValueError("repair reviewer provenance mismatch")
    validate_review(report, context)
    discoveries = []
    for item in report["discoveries"]:
        # Global normalized mechanism/task novelty; version-local UCB never carries old credit.
        key = sha256_json(
            {
                "task": item["task_id"],
                "type": item["discovery_type"],
                "mechanism": " ".join(item["mechanism"].lower().split()),
            }
        )
        all_seeds = {
            c["seed"] for c in observations["cells"] if c["task_id"] == item["task_id"]
        }
        if (
            item["repair_related"]
            and item["novel"]
            and len(item["seeds"]) >= min_replications
            and set(item["seeds"]) == all_seeds
            and key not in prior_keys
        ):
            discoveries.append(key)
    return {
        **base,
        "status": "discovery" if discoveries else "inconclusive",
        "reward": int(bool(discoveries)),
        "discovery_keys": sorted(set(discoveries)),
        "review": report,
    }
