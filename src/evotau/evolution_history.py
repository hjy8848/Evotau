"""Bounded metadata summaries; no old raw trajectory or Customer hidden script."""


def summarize_effects(entries, *, max_families=8):
    families = {}
    for entry in entries:
        e = entry["effect"]
        family = e["semantic_family"]
        row = families.setdefault(
            family,
            {
                "family": family,
                "attempts": 0,
                "successful_mechanisms": [],
                "rejected_mechanisms": [],
                "known_fixes": [],
                "known_regressions": [],
                "unresolved": [],
            },
        )
        row["attempts"] += 1
        key = "successful_mechanisms" if e["accepted"] else "rejected_mechanisms"
        row[key].append(
            {
                "mutation_id": e["mutation_id"],
                "type": e["mutation_type"],
                "reason": e["rejection_reason"],
            }
        )
        for dest, source in [
            ("known_fixes", "fail_to_pass"),
            ("known_regressions", "pass_to_fail"),
            ("unresolved", "fail_to_fail"),
        ]:
            row[dest] = sorted(set(row[dest]) | set(e[source]))
    return [families[k] for k in sorted(families)[-max_families:]]


def stagnation_state(generations, entries, patience):
    no_promotion = 0
    for g in reversed(generations):
        if g["service_phase"]["accepted"]:
            break
        no_promotion += 1
    recent = entries[-patience:]
    same_family = (
        len(recent) >= patience
        and len({e["mutation"]["semantic_family"] for e in recent}) == 1
    )
    persistent = bool(recent) and all(
        set(e["effect"]["fail_to_fail"]) == set(recent[0]["effect"]["fail_to_fail"])
        for e in recent
    )
    uncertainty = len(recent) >= patience and all(
        e.get("gate", {}).get("verdict") == "INCONCLUSIVE" for e in recent
    )
    return {
        "explore": no_promotion >= patience or same_family or uncertainty,
        "mode": "EXPLORE_ON_STAGNATION"
        if no_promotion >= patience or same_family or uncertainty
        else "REFINE",
        "no_promotion_generations": no_promotion,
        "repeated_family": same_family,
        "unchanged_failure_cluster": persistent,
        "uncertain_improvement": uncertainty,
        "avoid_families": sorted({e["mutation"]["semantic_family"] for e in recent})
        if same_family
        else [],
        "preferred_mutation_types": ["split", "delete"]
        if no_promotion >= patience
        else [],
    }
