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
  Accuracy appears only when the entire fixed E panel is complete. NO_OP skips
  the Service replay row.
- Execution state supplies running/paused/failed status, including runs launched
  outside RunManager. A saved failure takes precedence over a stale live status.
- Failure diagnostics link to the actual incomplete attempt and last persisted
  native turn. Native rollout does not persist live task/turn data on every turn;
  unavailable live fields are labelled as unrecorded, not inferred.
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

Console Python files are included in EvoTau's runtime source fingerprint. A web
change therefore changes that fingerprint even though it does not change evolution.
The stopped experiment's conditions and saved artifacts have not been rewritten.

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
and read-only polling. Browser checks cover desktop/narrow layouts, real archived
failure values, polling, preserved skill expansion and console errors.
