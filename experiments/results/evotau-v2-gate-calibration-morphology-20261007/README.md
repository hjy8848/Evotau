# V2 gate calibration: incomplete live morphology smoke

Code gates were fixed and covered by 189 passing offline tests before live execution. This is a **targeted diagnostic, not a formal experiment or effectiveness result**.

| Treatment | Completed | Native accuracy | Fixes/breaks |
|---|---:|---:|---|
| Empty memory | 12/12 | 10/12 = 83.33% | Baseline |
| Same guidance, render-all | 0/12 | Unavailable | Unavailable |
| Same guidance, narrowed activation | 0/12 | Unavailable | Unavailable |

Tasks: 22, 80, 98, 4, 21, 35; paired seeds 1 and 2. The baseline failures are 21/seed2 and 35/seed2, both max_steps. Tasks 22 and 80 pass both seeds, so the historical 2-fix/4-break failure morphology was not reproduced in this baseline. Seed1 is 6/6; seed2 is 4/6. Do not treat two seeds as two independent tasks.

Runtime/native evaluator: InferAI Qwen3.7-plus, thinking disabled, max_steps32, four episode workers, shared 30 requests/60.1s pacing, unlimited total request budget. Evolver: InferAI DeepSeek V4 Pro; thinking enabled and max_tokens65536 on wire. Configuration requests reasoning_effort high, but the captured wire omits reasoning_effort, so high effort is not independently established.

After baseline completion, the narrower-signature proposal returned HTTP200, finish_reason=stop, 93,502 prompt tokens and 10,848 completion tokens. Its visible JSON omits the final closing brace. The strict parser stopped before semantic validation, render-all/activation rollouts, or promotion. The reply is retained; it was not silently repaired or retried. The provider ledger records successful HTTP calls, while the research stage is correctly FAILED; these are distinct statuses.

A deliberate engineering interruption before any signature request switched the new diagnostic launcher from raw simulation objects to the existing observable trajectory projection. All 12 complete baseline episodes and both model probes were reused. The pause and correction are retained in diagnostic/engineering-interruption.json. Native runtime source remained frozen at 19f6882; launcher correction is 54c8c6e.

Total: **252 provider calls; 1,499,951 prompt tokens; 52,731 completion tokens; 692.97 seconds (11.55 minutes) from first request to failure, including the engineering pause**. 252/252 responses HTTP200. All 250 Qwen requests used disabled thinking; no reasoning tokens/content were observed. Reviewer/Judge calls are zero. See diagnostic-report.json for per-call usage and actual wire metadata.

Gate calibration: current Customer superiority is required on E; V and archived/native replay require preservation, with ties allowed. Small V3 explicitly uses finite_panel_paired: repeat paired seeds, no observed loss/new stuck/hard violation, no population-risk certificate. min_tasks8 is unchanged for task-block population inference. Screen protects passing task×seed cells, including those belonging to a target task. Archive dominance is restricted to matching evaluation scope and exact paired baseline/cells.

No V/H content was evaluated. No candidate was accepted. Activation efficacy and the complete V2 search remain untested live. Re-running the launcher is an explicit resume and would make a new proposal request; successful native episodes remain reusable under the frozen manifest.
