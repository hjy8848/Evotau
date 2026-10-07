# Skill Evolution V2: baseline and execution invariants

Baseline: 6fa200421dead77db1bac5e9694dd67825428778.

V1 flows config → frozen manifest → split-aware E/V loader → native runner → Customer proposal/selection → one Service proposal → E accuracy/V gate → immutable generation/checkpoint. Service overlays subclass the pinned LLMAgent; scenarios go only to the native UserSimulator and Customer Evolver. Native evaluator success remains authoritative. H objects are streamed only after evolution and fresh challenge generation.

Cache conditions are task ID, seed, Customer strategy ID, Service memory ID. A run manifest freezes models, model args, source fingerprints and panels. Per-condition locks prevent double dispatch; RequestBudget reservations prevent concurrent oversubscription. Failed attempts remain diagnostic and cannot enter fitness. Frozen proposal inputs are checked before reuse. Identical immutable bytes are reusable; differing bytes are rejected. Checkpoint publication is atomic.

V2 adds a separate orchestration path, preserving V1 schemas and defaults. Runtime memory contains only promoted skills. Candidate memory is used only for its explicit evaluation conditions; research archive/history never enters runtime. Activation accepts only projected observed messages plus trigger/signature catalog, excludes guidance from the selection catalog and excludes hidden metadata. Each response receives only selected guidance. Activation configuration is manifest-bound, hence within-run cache identity remains sufficient.

Each V2 stage is an immutable envelope bound to manifest, input hash and payload hash. Native episode cache handles interrupted panels. Stage artifacts persist before later stages run; a resume verifies the entire journal, checkpoint digest and committed generation chain. Search ancestry, rejected effects, summaries and statistical policy are saved outside runtime memory. Promotion uses native success, complete paired outcomes, no hard regressions, task-block statistics and frozen multiplicity correction. Formal gate is V; E-only is explicitly mechanism smoke. H never participates in diagnosis, archive, replay or activation selection.

Implementation verification is separate from research effectiveness. Deterministic integration is evidence of plumbing correctness only; no model-effectiveness claim follows.
