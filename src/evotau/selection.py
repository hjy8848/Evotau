"""Deterministic incumbent-preserving selection and strict confirmation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .records import CandidateEvaluation


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    incumbent_id: str
    selected_id: str
    evolved: bool
    reason: str
    discovery_scores: tuple[tuple[str, int], ...]
    confirmation_scores: tuple[tuple[str, int], ...] = ()


def select_customer(
    incumbent: CandidateEvaluation,
    candidates: tuple[CandidateEvaluation, ...],
    *,
    confirmation: Mapping[str, CandidateEvaluation] | None = None,
) -> SelectionDecision:
    """Select one strict discovery winner only if a fresh paired panel confirms it."""

    if incumbent.panel_name != "discovery":
        raise ValueError("incumbent evaluation must use the discovery panel")
    for candidate in candidates:
        if candidate.panel_name != "discovery":
            raise ValueError("all Customer candidates must use the same discovery panel")
        if candidate.task_coverage != incumbent.task_coverage:
            raise ValueError("incumbent and candidates must have equal task coverage")
        if _episode_keys(candidate) != _episode_keys(incumbent):
            raise ValueError("incumbent and candidates must use the same task/seed discovery panel")
    scores = ((incumbent.strategy_id, incumbent.fitness),) + tuple(
        (candidate.strategy_id, candidate.fitness) for candidate in candidates
    )
    # Stable ID tie-breaking makes proposal order irrelevant. The incumbent is
    # still the strict baseline and wins all ties.
    best_score = max((score for _, score in scores), default=incumbent.fitness)
    if best_score <= incumbent.fitness:
        return SelectionDecision(
            incumbent.strategy_id,
            incumbent.strategy_id,
            False,
            "no strict discovery improvement; incumbent retained",
            scores,
        )
    winner = min(
        (candidate for candidate in candidates if candidate.fitness == best_score),
        key=lambda candidate: candidate.strategy_id,
    )
    if confirmation is None or winner.strategy_id not in confirmation:
        return SelectionDecision(
            incumbent.strategy_id,
            incumbent.strategy_id,
            False,
            "strict discovery improvement lacks fresh paired confirmation",
            scores,
        )
    incumbent_confirm = confirmation.get(incumbent.strategy_id)
    challenger_confirm = confirmation[winner.strategy_id]
    if incumbent_confirm is None:
        return SelectionDecision(
            incumbent.strategy_id,
            incumbent.strategy_id,
            False,
            "confirmation must evaluate both incumbent and challenger",
            scores,
        )
    if incumbent_confirm.panel_name != "confirmation" or challenger_confirm.panel_name != "confirmation":
        raise ValueError("confirmation results must be marked with panel_name='confirmation'")
    if (incumbent_confirm.strategy_id != incumbent.strategy_id
            or challenger_confirm.strategy_id != winner.strategy_id):
        raise ValueError("confirmation results must match the incumbent and selected challenger IDs")
    if _episode_keys(incumbent_confirm) != _episode_keys(challenger_confirm):
        raise ValueError("incumbent and challenger must use the same confirmation task/seed panel")
    if _episode_keys(incumbent_confirm) & _episode_keys(incumbent):
        raise ValueError("confirmation must use a fresh task/seed panel")
    incumbent_confirm_fitness = _reproduced_task_count(incumbent, incumbent_confirm)
    challenger_confirm_fitness = _reproduced_task_count(winner, challenger_confirm)
    confirmation_scores = (
        (incumbent.strategy_id, incumbent_confirm_fitness),
        (winner.strategy_id, challenger_confirm_fitness),
    )
    if challenger_confirm_fitness <= incumbent_confirm_fitness:
        return SelectionDecision(
            incumbent.strategy_id,
            incumbent.strategy_id,
            False,
            "fresh paired confirmation did not strictly improve; incumbent retained",
            scores,
            confirmation_scores,
        )
    return SelectionDecision(
        incumbent.strategy_id,
        winner.strategy_id,
        True,
        "strict discovery and independent confirmation improvement",
        scores,
        confirmation_scores,
    )


def _episode_keys(evaluation: CandidateEvaluation) -> frozenset[tuple[str, int]]:
    keys = [(episode.task_id, episode.seed) for episode in evaluation.episodes]
    if len(keys) != len(set(keys)):
        raise ValueError("evaluation contains duplicate task/seed episodes")
    return frozenset(keys)


def _reproduced_task_count(discovery: CandidateEvaluation,
                           confirmation: CandidateEvaluation) -> int:
    """Count only discovery failures independently repeated in this fresh panel."""
    confirmation_by_id = {item.episode_id: item for item in confirmation.episodes}
    audited = dict(confirmation.failure_audit_refs)
    reproduced_tasks = set()
    for failure in discovery.verified_failures:
        replay = confirmation_by_id.get(failure.reproduction_episode_id)
        if replay is None or not replay.has_attributable_failure_candidate:
            continue
        if (replay.task_id, replay.customer_strategy_id, replay.service_strategy_id) != (
            failure.task_id, failure.customer_strategy_id, failure.service_strategy_id,
        ) or audited.get(replay.episode_id) != failure.reproduction_verification_ref:
            continue
        if (replay.policy_rule_id, replay.workflow_stage, replay.mistake_type) != (
            failure.policy_ref, failure.signature.workflow_stage, failure.signature.mistake_type,
        ):
            continue
        source_seeds = {item.seed for item in discovery.episodes if item.episode_id == failure.episode_id}
        if not source_seeds or replay.seed in source_seeds:
            continue
        reproduced_tasks.add(failure.task_id)
    return len(reproduced_tasks)
