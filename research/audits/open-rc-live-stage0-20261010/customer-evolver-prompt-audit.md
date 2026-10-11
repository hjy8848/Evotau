# Customer Evolver prompt audit — 2026-10-11

Read-only source and saved-request audit; no API calls, prompt edits, or scoring changes.

## Verified findings

1. Actual `evotau_customer_skill_evolver_v1` request uses CUSTOMER_SKILL_PROMPT. It asks for observable weaknesses, but lists conditional confirmation, cost explanations and correction as examples. It does not explicitly distinguish a challenge from procedural assistance or ask the model to explain why its procedure will expose a failure rather than prevent one. This is a plausible generation bias, not proof of a general model deficiency.
2. CustomerSkill INVARIANTS additionally requires adequate cost explanations before constrained authorization. Preservation of facts/consent is necessary; this wording also makes a price-checking procedure attractive. Do not delete conditional authorization safeguards merely to increase failures.
3. The actual request covers all 10 E outcomes, but expands only tasks 39, 34, 15 (two successes and one failure). Task 12 is an overview rather than an expanded trajectory. Complexity ranking is deterministic and does not establish weakness or mechanism relevance. No V/H contamination was observed in this request.
4. requested_candidates=1, customer_attempt_history=[], current Service memory empty. This is one initial proposal, not evidence of iterative search failure. Incumbent accuracy is supplied; rollout selection correctly requires lower accuracy.
5. Candidate requests complete itemized costs and explicit authorization. Actual native E outcome: 6/10 -> 8/10, no new failures, recoveries at tasks 12 and 34. On task 12, Customer discloses $650 upfront and requests quotes, avoiding the baseline $1200 write. This supports an assistance interpretation, but independently generated conversations and one seed prevent causal attribution.
6. Four trajectory validation cells are uncertain because evidence indices do not cite original Customer messages. This affects eligibility, independently of the absence of new failures; it must not be described as the cause of the 80% success rate.

## Recommended scope for a separately frozen follow-up

- Clarify objective and require a testable hypothesis identifying the Service decision/dependency under pressure, why a competent Service can still succeed, and why the procedure does not simply supply the missing solution/checklist. Keep open-ended mechanisms and all legality constraints.
- Distinguish task-faithful necessary cooperation from volunteered procedural coaching. Do not require withholding necessary facts, concealing constraints, or unauthorized actions.
- Record trigger exposure and execution separately from native fitness; do not use exposure annotations to manufacture scores or promotion.
- Audit validator evidence-index alignment as a separate engineering contract issue before claiming valid new discoveries.

No proposed prompt improvement is empirically verified. No changes to prompt, search budget, gates, native runtime, or provider were made.
