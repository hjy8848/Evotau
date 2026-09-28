"""Strict failure promotion: heuristic signals never become fitness by themselves."""

from __future__ import annotations

from dataclasses import dataclass

from .records import (
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
) -> AttributionDecision:
    """Promote only after an explicit independent verifier accepts the trace.

    The reference is expected to identify a human/protocol audit artifact. A raw
    reviewer label or task failure alone is deliberately insufficient.
    """

    if not episode.has_attributable_failure_candidate:
        return AttributionDecision(episode.episode_id, False, "episode lacks attributable evidence")
    if not independent_verification_ref or not independent_verification_ref.strip():
        return AttributionDecision(episode.episode_id, False, "independent verification is missing")
    assert episode.policy_rule_id and episode.mistake_type and episode.workflow_stage
    if not is_mvp_failure_signature(FailureSignature(
        "retail", episode.workflow_stage, episode.policy_rule_id, episode.mistake_type,
    )):
        return AttributionDecision(episode.episode_id, False, "failure type is outside the frozen MVP taxonomy")
    failure = FailureRecord.verify(
        episode,
        generation=generation,
        verifier=independent_verification_ref.strip(),
    )
    return AttributionDecision(episode.episode_id, True, "independently verified", failure)
