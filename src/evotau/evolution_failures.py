"""Paired native outcomes and evolution-only evidence. Unknown is never failure."""

from dataclasses import asdict, dataclass
from statistics import mean

from .records import EpisodeStatus


@dataclass(frozen=True, slots=True)
class MutationEffect:
    mutation_id: str
    generation: int
    parent_memory_id: str
    proposed_memory_id: str
    semantic_family: str
    root_cause_cluster_id: str
    mutation_type: str
    old_accuracy: float | None
    candidate_accuracy: float | None
    fail_to_pass: tuple[str, ...]
    pass_to_fail: tuple[str, ...]
    fail_to_fail: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    helpfulness: float | None
    harmfulness: float | None
    old_stuck_rate: float | None
    new_stuck_rate: float | None
    step_delta: float | None
    token_delta: float | None
    accepted: bool
    rejection_reason: str | None
    parent_mutation_ids: tuple[str, ...] = ()

    def to_dict(self):
        value = asdict(self)
        for key, field in value.items():
            if isinstance(field, tuple):
                value[key] = list(field)
        return value


def observable_outcome(record):
    return record.task_success if record.status == EpisodeStatus.COMPLETE else None


def is_stuck(record):
    return "max_steps" in (record.termination_reason or "").lower()


def episode_metrics(records):
    known = [r for r in records if observable_outcome(r) is not None]
    steps = sorted(r.total_steps for r in records if r.total_steps is not None)
    tokens = [
        r.prompt_tokens + r.completion_tokens
        for r in records
        if r.prompt_tokens is not None and r.completion_tokens is not None
    ]
    return {
        "accuracy": mean(observable_outcome(r) for r in known)
        if len(known) == len(records) and records
        else None,
        "unknown_count": len(records) - len(known),
        "stuck_rate": mean(is_stuck(r) for r in records) if records else None,
        "max_steps_count": sum(is_stuck(r) for r in records),
        "mean_steps": mean(steps) if len(steps) == len(records) and steps else None,
        "p95_steps": steps[min(len(steps) - 1, int(0.95 * len(steps)))]
        if steps
        else None,
        "mean_tool_calls": mean(r.tool_calls for r in records) if records else None,
        "mean_tokens": mean(tokens) if len(tokens) == len(records) and tokens else None,
        "hard_violations": sum(r.hard_policy_protocol_violations for r in records),
        "mixed_text_tool_call": sum(r.mixed_text_tool_call_messages for r in records),
    }


def paired_failure_matrix(old, new, *, generation, candidate_id, prior=()):
    left = {(r.task_id, r.seed): r for r in old}
    right = {(r.task_id, r.seed): r for r in new}
    if len(left) != len(old) or len(right) != len(new) or set(left) != set(right):
        raise ValueError(
            "paired evaluation requires identical unique task × seed cells"
        )
    rows = []
    for key, before in left.items():
        after = right[key]
        a, b = observable_outcome(before), observable_outcome(after)
        status = (
            "UNCERTAIN"
            if a is None or b is None
            else {
                (True, True): "STABLE_PASS",
                (False, True): "FIXED",
                (True, False): "BROKEN",
                (False, False): "PERSISTENT_FAIL",
            }[(a, b)]
        )
        history = [r for r in prior if r["task_id"] == key[0]]
        cross = (
            "FIXED_HISTORY"
            if b is True
            and any(
                r["status"] in ("FIXED", "BROKEN", "PERSISTENT_FAIL") for r in history
            )
            else "PERSISTENT"
            if b is False
            and history
            and all(r["new_success"] is False for r in history)
            else "RECURRING"
            if b is False and any(r["new_success"] is False for r in history)
            else "NEW_FAILURE"
            if b is False
            else None
        )
        rows.append(
            {
                "task_id": key[0],
                "seed": key[1],
                "generation": generation,
                "candidate_id": candidate_id,
                "status": status,
                "cross_generation": cross,
                "old_success": a,
                "new_success": b,
                "old_episode_id": before.episode_id,
                "episode_id": after.episode_id,
                "termination_reason": after.termination_reason,
                "max_steps": is_stuck(after),
                "total_steps": after.total_steps,
                "tool_calls": after.tool_calls,
                "tool_errors": after.tool_errors,
                "protocol_violations": after.hard_policy_protocol_violations,
                "mixed_text_tool_call": after.mixed_text_tool_call_messages,
                "prompt_tokens": after.prompt_tokens,
                "completion_tokens": after.completion_tokens,
                "activated_skill_ids": list(after.activated_skill_ids),
            }
        )
    return rows


def paired_effect(
    old,
    new,
    mutation,
    *,
    generation,
    mutation_id,
    parent_memory_id,
    proposed_memory_id,
    accepted=False,
    reason=None,
    parent_ids=(),
):
    matrix = paired_failure_matrix(
        old, new, generation=generation, candidate_id=mutation_id
    )
    grouped = {
        key: tuple(dict.fromkeys(r["task_id"] for r in matrix if r["status"] == label))
        for key, label in [
            ("fail_to_pass", "FIXED"),
            ("pass_to_fail", "BROKEN"),
            ("fail_to_fail", "PERSISTENT_FAIL"),
            ("pass_to_pass", "STABLE_PASS"),
        ]
    }
    known = [r for r in matrix if r["status"] != "UNCERTAIN"]
    failures = sum(r["old_success"] is False for r in known)
    successes = sum(r["old_success"] is True for r in known)
    a, b = episode_metrics(old), episode_metrics(new)

    def delta(key):
        return None if a[key] is None or b[key] is None else b[key] - a[key]

    return MutationEffect(
        mutation_id,
        generation,
        parent_memory_id,
        proposed_memory_id,
        mutation["semantic_family"],
        mutation["target_cluster_id"],
        mutation["operation"],
        a["accuracy"],
        b["accuracy"],
        **grouped,
        helpfulness=sum(r["status"] == "FIXED" for r in known) / failures
        if failures
        else None,
        harmfulness=sum(r["status"] == "BROKEN" for r in known) / successes
        if successes
        else None,
        old_stuck_rate=a["stuck_rate"],
        new_stuck_rate=b["stuck_rate"],
        step_delta=delta("mean_steps"),
        token_delta=delta("mean_tokens"),
        accepted=accepted,
        rejection_reason=reason,
        parent_mutation_ids=tuple(parent_ids),
    )
