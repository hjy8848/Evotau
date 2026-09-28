"""Strict failure promotion: heuristic signals never become fitness by themselves."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .records import (
    CandidateEvaluation,
    EpisodeRecord,
    FailureRecord,
    FailureSignature,
    is_mvp_failure_signature,
)


@dataclass(frozen=True, slots=True)
class AttributionDecision:
    episode_id: str
    promoted: bool
    reason: str
    failure: FailureRecord | None = None


def promote_verified_failure(
    episode: EpisodeRecord,
    *,
    generation: int,
    independent_verification_ref: str | None,
    reproduction_episode: EpisodeRecord | None = None,
    reproduction_verification_ref: str | None = None,
) -> AttributionDecision:
    """Promote only after independent trace audits and a fresh-seed rerun agree.

    Both references identify separate human/protocol audit artifacts. A second
    judgment of the original trace is not a substitute for ``reproduction_episode``.
    """

    if not episode.has_attributable_failure_candidate:
        return AttributionDecision(episode.episode_id, False, "episode lacks attributable evidence")
    if (not isinstance(independent_verification_ref, str)
            or not independent_verification_ref.strip()):
        return AttributionDecision(episode.episode_id, False, "independent verification is missing")
    if (reproduction_episode is None or not isinstance(reproduction_verification_ref, str)
            or not reproduction_verification_ref.strip()):
        return AttributionDecision(episode.episode_id, False, "strategy-level failure reproduction is missing")
    if not reproduction_episode.has_attributable_failure_candidate:
        return AttributionDecision(episode.episode_id, False, "reproduction episode has no attributable failure")
    if (episode.task_id, episode.customer_strategy_id, episode.service_strategy_id) != (
        reproduction_episode.task_id,
        reproduction_episode.customer_strategy_id,
        reproduction_episode.service_strategy_id,
    ):
        return AttributionDecision(episode.episode_id, False, "reproduction changed task or strategy")
    if episode.seed == reproduction_episode.seed:
        return AttributionDecision(episode.episode_id, False, "reproduction did not use a fresh seed")
    if (episode.policy_rule_id, episode.workflow_stage, episode.mistake_type) != (
        reproduction_episode.policy_rule_id,
        reproduction_episode.workflow_stage,
        reproduction_episode.mistake_type,
    ):
        return AttributionDecision(episode.episode_id, False, "reproduction did not match the exact failure signature")
    assert episode.policy_rule_id and episode.mistake_type and episode.workflow_stage
    if not is_mvp_failure_signature(FailureSignature(
        "retail", episode.workflow_stage, episode.policy_rule_id, episode.mistake_type,
    )):
        return AttributionDecision(episode.episode_id, False, "failure type is outside the frozen MVP taxonomy")
    failure = FailureRecord.verify(
        episode,
        reproduction_episode=reproduction_episode,
        generation=generation,
        verifier=independent_verification_ref.strip(),
        reproduction_verifier=reproduction_verification_ref.strip(),
    )
    return AttributionDecision(episode.episode_id, True, "independently audited and reproduced", failure)


def confirm_failure_reproductions(
    discovery: CandidateEvaluation,
    confirmation: CandidateEvaluation,
    *,
    generation: int,
) -> tuple[CandidateEvaluation, CandidateEvaluation, tuple[FailureRecord, ...]]:
    """Cross-link independently audited, same-signature failures across fresh runs.

    One deterministic occurrence per ``(task, signature)`` is retained in the
    discovery panel, with the confirmation episode recorded as reproduction
    evidence. Confirmation episodes remain observations, not duplicate failures.
    Unmatched signals remain unpromoted.
    """
    if discovery.panel_name != "discovery" or confirmation.panel_name != "confirmation":
        raise ValueError("failure reproduction requires discovery and confirmation panels")
    if discovery.strategy_id != confirmation.strategy_id:
        raise ValueError("failure reproduction must keep the Customer strategy fixed")
    if _panel_keys(discovery) != _panel_keys(confirmation):
        raise ValueError("failure reproduction panels must use the same task set")
    discovery_keys = {(item.task_id, item.seed) for item in discovery.episodes}
    confirmation_keys = {(item.task_id, item.seed) for item in confirmation.episodes}
    if discovery_keys & confirmation_keys:
        raise ValueError("failure reproduction confirmation must use fresh task/seed pairs")
    discovery_audits = dict(discovery.failure_audit_refs)
    confirmation_audits = dict(confirmation.failure_audit_refs)
    discovery_candidates = _failure_candidates(discovery)
    confirmation_candidates = _failure_candidates(confirmation)
    matched = sorted(set(discovery_candidates) & set(confirmation_candidates))
    discovery_failures: list[FailureRecord] = []
    for key in matched:
        source = discovery_candidates[key]
        replay = confirmation_candidates[key]
        source_ref = discovery_audits.get(source.episode_id)
        replay_ref = confirmation_audits.get(replay.episode_id)
        if not source_ref or not replay_ref:
            continue
        discovery_failures.append(FailureRecord.verify(
            source, reproduction_episode=replay, generation=generation,
            verifier=source_ref, reproduction_verifier=replay_ref,
        ))
    if not discovery_failures:
        return discovery, confirmation, ()

    discovery_replications = tuple(
        failure_episode for failure_episode in confirmation.episodes
        if any(failure_episode.episode_id == failure.reproduction_episode_id
               for failure in discovery_failures)
    )
    discovery_replication_ids = {item.episode_id for item in discovery_replications}
    discovery_refs = tuple(discovery.failure_audit_refs) + tuple(
        item for item in confirmation.failure_audit_refs if item[0] in discovery_replication_ids
    )
    discovery_confirmed = replace(
        discovery,
        verified_failures=(*discovery.verified_failures, *discovery_failures),
        failure_audit_refs=discovery_refs,
        replication_episodes=discovery_replications,
    )
    return (
        discovery_confirmed,
        confirmation,
        tuple(discovery_failures),
    )


def _failure_candidates(evaluation: CandidateEvaluation) -> dict[tuple[str, str], EpisodeRecord]:
    output: dict[tuple[str, str], EpisodeRecord] = {}
    for episode in sorted(evaluation.episodes, key=lambda item: (item.seed, item.episode_id)):
        if not episode.has_attributable_failure_candidate:
            continue
        signature = FailureSignature(
            "retail", episode.workflow_stage or "", episode.policy_rule_id or "", episode.mistake_type or "",
        )
        if is_mvp_failure_signature(signature):
            output.setdefault((episode.task_id, signature.key), episode)
    return output


def _panel_keys(evaluation: CandidateEvaluation) -> frozenset[str]:
    return frozenset(episode.task_id for episode in evaluation.episodes)
