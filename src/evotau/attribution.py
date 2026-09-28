"""Strict failure promotion: heuristic signals never become fitness by themselves."""

from __future__ import annotations

from dataclasses import dataclass

from .records import EpisodeRecord, FailureRecord


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
    failure = FailureRecord.verify(
        episode,
        generation=generation,
        verifier=independent_verification_ref.strip(),
    )
    return AttributionDecision(episode.episode_id, True, "independently verified", failure)
