# Evolver format recovery v4

This is an offline-verified reliability protocol, not evidence of Airline Skill effectiveness.
No real provider requests were made during implementation or verification.

## Frozen identity and compatibility

`analyst_skill_recovery_v4` opts into `evolver_recovery.protocol_version: format_recovery_v1`
and `max_format_recoveries: 2`. Valid configured extra-attempt counts are 0–2. These are
per-logical-call recovery limits, not experiment request/token/context budget caps.
Legacy canonical policies serialize without any recovery field and continue using the original
Dedup prompt and fail-closed JSON behavior. The manual `release_recovery.py` authorization and
accounting reconciliation APIs remain unchanged. An operator authorization cannot silently
extend an automatic logical call's frozen attempt allowance; such a conflict stops explicitly.

New prepared config: `configs/airline-analyst-recovery-v4-hybrid-seeds-p2.yaml`.
It retains the preceding hybrid config's E10/V20/H20 IDs, native runtime, model roles,
Screen, Skill Validator, V Gate, task-block bootstrap, Bonferroni correction, independent
seed schedules, two workers and max_steps=200. Only identity, recovery policy and preparatory
launch metadata differ. Output/checkpoint directories are new. Neither old failed run is
modified or resumed in place; no episodes or scores are automatically imported.

## State machine and scope

The centralized recovery controller wraps Failure Analyst, Mechanism Deduplication and Assigned
Skill Mutator. Each invocation follows:

1. Guard Service context; bind method, frozen context, model/args, Manifest and recovery policy.
2. Freeze a dispatch intent and attempt identity in an independent EvolutionJournal.
3. Execute the original provider/budget chain; retain complete visible text and metadata.
4. Execute the original JSON, schema AND business/evidence checks.
5. Original success: `VALID`. Later success: `RECOVERED`.
6. JSON/schema failure: immutable `REJECTED` attempt, up to two additional generations using
   identical model/args/context and a numbered formatting-contract reminder.
7. Exhaustion: logical call `RECOVERY_EXHAUSTED`; the stage uses the following frozen response.

No local comma/bracket repair, fragment concatenation, semantic transformation, new evidence,
provider change, structured-output capability assumption or unbounded sampling is used.
A regenerated response is not claimed to be semantically equivalent to malformed original text.
The assigned Mutator still must copy the assigned root cause/target set and cite only supplied
original messages. Invalid task labels, fabricated hashes/IDs and missing pair comparisons
cannot pass merely because a recovery JSON parses.

| Stage exhausted | Stage result and next step |
| --- | --- |
| Analyst | `REJECTED`, unavailable proposal with zero hypotheses; retain current Service and continue generation |
| Dedup | `DEGRADED`; choose at most ONE already evidence-validated `skill` hypothesis, descending retained evidence count and stable input order; defer all others |
| Dedup with only uncertain/not_skill hypotheses | `DEGRADED` / `NO_OP`; do not label an uncertain case repairable |
| Mutator | `REJECTED`; retain current Service and continue other assigned candidates |

A legal Skill/NO_OP retains the complete downstream Validator/Screen/E/V protocol. There is
no forced promotion. A valid-JSON semantic validator refusal remains an ordinary rejection.
Customer generation/semantic validation, Skill semantic validation and crossover have not acquired
new format retry policies in this version; their previous safeguards remain intact.

## Errors that stop

Transport failures (including timeout/429/401/403), request-budget exhaustion, native episode
execution errors, pause signals, configuration/provenance drift, journal/checkpoint/cache digest
errors and explicit hidden/V/H payload contamination are NOT converted into task failures,
recovery exhaustion or candidate fitness. They propagate; execution evidence is retained.
Controller infrastructure events are labeled `FAILED_INFRASTRUCTURE` and re-raised.
No automatic network retries were added: the absence of billing/submission side effects is not
reliably established for current gateways. This intentionally leaves transport recovery as a
separate operator-controlled task.

The Service guard rejects explicit hidden/evaluation fields and V/H panel payloads; existing
context builders/data sealing and exact evidence registries remain the primary data boundaries.
This is not a claim to detect arbitrary private facts embedded in free-form model prose.

## Persistence and exact resume

`format-recovery/evolution-v2/` uses the existing immutable atomic EvolutionJournal publication,
Manifest/input/output/envelope digests and run-level writer lock. Every attempt records its
actual provider-call ID, parent request, attempt index, source input/output/evidence digests,
provider ledger digest, elapsed time and usage. Failures and successes are never overwritten.

A completed logical result (including exhaustion) is reused exactly. A completed provider
response interrupted before attempt/stage publication is reused and fully revalidated.
A matching submitted input without a durable response, or a prior transport failure, becomes
`UnknownRequestState`; no blind request is issued. Original E episodes and successful stages
continue to use their existing journal/cache binding. Changed configuration requires a new run.
Manual retry cannot bypass the protocol counter. Existing interrupted-accounting reconciliation
is an explicit operator operation, never an automatic assumption of success or fabricated tokens.

## Accounting and interpretation

All attempts use the same RequestBudget and per-call provider logs, so total attempts, tokens,
model/role usage and elapsed time include recovery. The final report additionally contains
`evolver_recovery` with logical/original-valid/recovered/exhausted counts, recovery success rate,
extra request/token/elapsed usage and degraded stage count. Stage artifacts retain the recovery
outcome and references. API HTTP success and structurally valid original output are different
metrics; use provider_usage for the former. Skill generation, E fixes/regressions, V verdicts and
H results remain in their original report fields. Extra sampling must not be credited entirely
to the Analyst prompt. Identical frozen Dedup inputs across generations may reuse a completed
exhausted result rather than paying for three identical failures again.

## Offline verification

`tests/fixtures/mechanism-dedup-incident-visible.json` is an exact copy of the real incident's
visible-completion artifact, including its original recorded hash. Tests cover concatenated
objects, missing closure, recovery success/exhaustion, conservative selection and NO_OP,
schema/evidence failures, metadata/accounting, unknown execution, integrity failure, interrupted
publication, G2 continuation and exact resume. An E10 integration simulation demonstrates that
the incident proceeds to a single legal Mutator after degradation and no completed E condition
is dispatched twice. A separate recovered valid mutation test reaches Screen, Full-E and V Gate.
Existing legacy/retail/native-runtime regression tests remain in the full suite.

```sh
PYTHONPATH=src TAU2_DATA_DIR=/Users/spring/.cache/evotau/tau2-data-b7ea9074 \
  /Users/spring/RSI/Evotau/.venv/bin/python -m pytest -q
/Users/spring/RSI/Evotau/.venv/bin/ruff check src tests
git diff --check
```

Preparation-only command (no `--execute`, no credentials, no paid requests):

```sh
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python \
  experiments/execution/run-airline-gateway.py --mode formal \
  --config configs/airline-analyst-recovery-v4-hybrid-seeds-p2.yaml \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074
```

Future launch/resume uses this SAME new config/command plus `--execute
--approve-unbounded-requests` and an independent pause-file path. It has NOT been executed.
A/A remains uncalibrated as explicitly recorded in the original protocol. Actual live recovery
success rates, network behavior and research effectiveness remain unverified.
