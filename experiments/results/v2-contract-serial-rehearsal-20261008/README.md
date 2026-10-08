# Explicitly authorized serial E3 rehearsal

The user requested serial execution after the two-worker Qwen HTTP429 failure.
This independent config changes episode concurrency from2 to1. Models, native
policy/tools/backend/evaluator, V2 prompts, E3 tasks98/8/66, seeds1/2, fitness seed1,
max_steps32 and evolution/gate logic stay unchanged. No old score, proposal,
episode or stage was imported. V/H were disabled; no held-out content was loaded.
Config dry validation passed. Runtime source fingerprint stays unchanged.

**The generation completed:**62 requests, all HTTP200, no429 or other provider
errors. Qwen59 calls, GPT6 Evolver3 calls. Observed peak episodes1 and HTTP
requests1; all59 Qwen requests sent thinking disabled and responses reported no
reasoning content/tokens. GPT uses the existing InferAI Responses route and GPT
credential group, reasoning.effort=high, no temperature. No automatic retries or
model fallback. A successful serial run does not prove which provider limit caused
the earlier failure; availability and load may also have changed.

Three native episodes: task98 fail, task8 fail, task66 pass. Initial/selected/final
accuracy is1/3. This differs from the previous baseline; old scores are not used as
this run's fitness and the difference is not evidence of a Service regression.
Customer Validator rejected the generated challenge for inventing a PayPal payment
method not supplied in one scenario. Its fitness is null, not a fabricated zero.

Diagnoser returned legal surfaces tool_boundary, stochastic_or_weak and
runtime_protocol, each with an empty mutation array. The generation therefore
produced no Service skill proposal and made no Service update. The diagnostic
claims are model inferences, not established native bugs. No benchmark component
was changed. **Native Skill Validator/Screen/Gate were not reached.** The research
repair effect and held-out generalization remain unverified; this completed run
must not be called a successful live candidate-Gate canary.

Reported usage:418,063 prompt tokens,17,732 completion tokens; all calls report
usage. Initial execution467.95 seconds (about7min48s). Completed replay adds zero
requests;26 completed episode/journal artifacts remain byte-identical. Reviewer,
Customer Judge and Service Judge calls:0.

This run's cap524 uses the previous allowance's remaining budget after76 prior
requests. Aggregate spent138/600;462 remain. No expensive formal E20/G2 run was
launched. Resuming a completed run issues no new calls; producing new candidates
would require a separately authorized experiment/config rather than modifying
this frozen run.

See summary.json, config.yaml, immutable native-run artifacts and sha256-index.json.
