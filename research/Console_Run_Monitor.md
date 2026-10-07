# Read-only alternating run monitor

The run detail page renders a compact monitor from saved artifacts and polls
`GET /runs/{run_id}/progress` every five seconds while the tab is visible. The
endpoint and page never launch or resume a run, modify fitness, or call a provider.
Polling errors retain the previous snapshot with a visible stale-data notice.

## Evidence and display rules

- The frozen manifest supplies E size, planned generations, worker limit and
  runtime source commit. The commit displayed is not the current Console commit.
- Checkpoint commits supply the active Customer and Service SkillMemory. Pending
  proposals do not become active strategies. Before the first commit, the initial
  checkpoint strategies are displayed.
- Manifest-bound stage/proposal artifacts supply the current phase. Panel
  references count complete, boolean-evaluated episodes, including explicit reuse.
  Fitness accuracy appears only when the entire fixed E panel is complete. Partial
  panels also display observed successes/completions under **PROVISIONAL / Not
  fitness**; these values never enter selection or acceptance. NO_OP skips
  the Service replay row.
- Execution state supplies running/paused/failed status, including runs launched
  outside RunManager. A saved failure takes precedence over a stale live status.
- Pending Customer proposals display their actual PromptStrategy text. Pending
  Service mutations display ADD/UPDATE/NO_OP, target ID, trigger, guidance and
  analysis. They remain separate from active committed strategies.
- Per-attempt `episodes/<attempt>/active-episode.json` publishes task/panel, last
  native message turn, role/tool name, next native role, activity and timestamp.
  An instance-local wrapper observes the pinned Orchestrator's `step()` before
  and after execution, delegates its original return/exception and restores the
  original method. It does not store message bodies, tool arguments, scenario,
  reasoning or hidden truth. Atomic writes isolate concurrent workers; telemetry
  failure is best-effort and cannot fail a native rollout. Terminal attempts are
  marked complete/failed; cache hits do not create a new attempt.
- Legacy failed attempts still link to their partial trace and last recorded turn.
  Legacy running attempts without telemetry retain explicit unavailable fields.
- Paired transitions show P→F/F→P/P→P/F→F counts and task IDs for matched, complete
  native episodes with the same task and seed, including committed generation
  history. Partial pair coverage is explicitly provisional; NO_OP does not imply
  a replayed Service repair.
- Failure cards show stage, original diagnostic type, task/panel, checkpoint
  preservation and current/frozen-source resume requirements. These observations
  do not grant automatic resume approval or promise provider recovery.
- Continuation separates imported/new complete episodes, imported/new incomplete
  attempts and reused panel references. References and attempts are different
  units; missing source inventories remain unavailable.
- `api-usage-live.json` supplies the global provider call count. Wire observations
  supply HTTP status counts, Flash thinking-disabled coverage, latest call latency
  and observed peak request concurrency. Missing wire observations stay missing;
  configured worker limits are not presented as observed concurrency.
- Sealed held-out phases hide task identities, traces and wire observations.
  Unsafe paths, mismatched manifest/generation bindings and malformed observations
  fail closed. One unfinished final JSONL append is retried on the next poll.
- Server-rendered values are escaped; refreshed values use DOM text nodes. The
  existing conversation/tool trace views remain available through episode links.

## Frozen runtime for existing experiments

Legacy source fingerprints include Console Python. Their hashing rules and frozen
manifests are unchanged. The stopped experiment's conditions and saved artifacts
have not been rewritten.

New alternating manifests use `evotau.source_scope: runtime-v2` and record
`runtime_source_sha256` (also retained as `source_sha256` for existing readers).
That hash covers runtime Python and `pyproject.toml`, excluding `web/**` and the
best-effort `observability.py`. A separate `console_source_sha256` covers Console
Python/templates/JS/CSS and the observer, and is recorded per execution invocation
in `run-execution-state.json`, outside the frozen execution identity. Web-only
edits can bind the same new frozen runtime manifest; actual runtime changes fail
binding. Legacy experiments require their frozen executable snapshot, rather than
an in-place migration to the new hash rules. Legacy Phase 0 provenance is retained.

Before this Console change, the source at `db3268911cfa1f04cf8bbe80927cf7befc1dc50e`
was preserved locally at:

`/Users/spring/.cache/evotau/frozen-runtime-965497fcfd0260fd`

Its verified source SHA-256 is:

`965497fcfd0260fdf20cc2525fecc024d8853bd82d1baa44730f1b0ce3902f3e`

The same source-only snapshot is recoverable with `git archive` of that commit's
`src` and `pyproject.toml`. If a later authorized resume uses this snapshot, import
the frozen `src` rather than the updated Console source, and keep the original
absolute config/output/checkpoint paths. Do not use an old launcher that inserts
the updated checkout's `src` ahead of the snapshot. No resume is performed by this
monitor or by the UI verification workflow.

## Verification

`tests/test_run_progress.py` covers failed/running/paused/completed states, partial
fitness, reuse, active initial/committed memory, NO_OP, missing/pending provider
observations, malformed latency, manifest/generation mismatch, sealed H, symlinks
and read-only polling. `tests/test_observability.py` executes actual pinned native
steps without providers, checks unchanged messages/exceptions, concurrent files,
and independent source hashes/resume binding. Native runner tests check terminal
telemetry and cache reuse. Browser checks cover desktop/narrow layouts, real
archived proposals/pairs/failures, polling, preserved expansion and console errors.
