# Offline Service Diagnoser evidence compression

No provider requests, native episodes, H reads or experiment resumes were performed.

The failed InferAI V4 Pro output64k diagnosis supplied six complete E trajectories.
The archived reconstruction replaces only the Diagnoser's trajectory input with all
20 task×seed overviews and four deterministic detailed cases: failed tasks 35, 54,
98 and passing control 21. Passing controls rank by overlap of observed tool names,
then task ID and seed; this is structural similarity, not a verified causal match.
All results remain native, including all 17 passes and three failures.

Measured system prompt + context: **48,697 → 30,084 proxy tokens (38.22% smaller)**.
The provider previously reported 50,723 input tokens for the original request;
cl100k_base is a reproducible local proxy, not DeepSeek's billed tokenizer.
Overviews occupy 7,130 proxy tokens, detailed evidence 16,259. Other unchanged
context, including policy and outcome records, accounts for the remainder.

Selected cases retain all user messages, mutating tool arguments, explicit tool
errors and adjacent action results. Bulky action results use explicit exact-field
projection; read-only results may have marked prefix excerpts and optional
messages may be omitted. Projection indices and original digests link every
retained message to immutable raw trajectories. No root cause is synthesized.
All task IDs are legal references under the existing strict Diagnoser schema,
but overview-only evidence cannot establish a causal mechanism on its own.
Service metadata is whitelisted: no task description, user scenario, user tools,
Customer strategy or extra task metadata is forwarded. Facts actually disclosed
in dialogue remain available as observed evidence.

Customer compression and Validator inputs are unchanged. Mutator context retains
its existing raw relevant cases and controls. Selection, native evaluation, E/V/H
panels, budgets, cache keys and promotion rules are unchanged. The new Diagnoser
input has a different journal/request hash and the runtime fingerprint changes;
no old frozen run has been silently resumed or rewritten.

`measure.py` verifies that the six original Diagnoser trajectories and all 20
outcomes match the archived E20 evidence before reconstructing the context. It
checks original hashes, exact messages/fields/excerpts, required evidence coverage
and source immutability. `measurement.json` records per-cell checks;
`compressed-input.json` is an offline comparison artifact, not a provider response.

This proves evidence preservation and integration offline. Diagnosis quality,
provider latency and successful live response remain unverified. Total native
rollout cost is unchanged; this is not a claim of a 38% experiment cost reduction.

Reproduce from the repository root with the pinned tau data environment:

```sh
TAU2_DATA_DIR=/Users/spring/.cache/evotau/tau2-data-b7ea9074 PYTHONPATH=src \
/Users/spring/RSI/Evotau/.venv/bin/python \
research/audits/diagnoser-context-compression-20261008/measure.py
```

Verification: 47 targeted tests passed (one existing tau2 audioop deprecation
warning); Ruff and git diff whitespace checks passed. Deterministic V2 tests
exercise diagnosis, mutation, screen, gate, archive and exact resume.
