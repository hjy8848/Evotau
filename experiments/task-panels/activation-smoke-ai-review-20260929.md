# Activation-smoke task review — AI pre-review

**Status: recommendation only; researcher confirmation pending. This file is not the activation runner’s review artifact.**

I checked candidate scenarios, evaluation criteria, the pinned Retail policy/tools, task hashes, and the reference actions against an isolated copy of the pinned database. No provider requests were made for this review.

Pinned source: `sierra-research/tau2-bench` `b7ea9074c1cba482b30687fecdb5c8425fd6f619`, package `1.0.1`. Official split has 74 train / 40 test tasks; no test task was loaded.

## Recommended panel

`E = 16, 22, 29, 35, 37, 52, 72, 73, 78, 88`; `V = 93`; `H = sealed / empty`.

This panel spans cancel/return/refund, profile and order address changes, delivered exchanges, pending item changes, and a conditional product/cancellation workflow. The 45 pairwise comparisons are recorded in the JSON and remain for researcher confirmation.

## Candidate decisions

| Task | AI recommendation | Evidence / reason |
|---:|---|---|
| 16 | eligible_recommended_selected | all 9 reference actions succeeded; The source provides name+ZIP for required authentication. In a fresh copy of the pinned Retail DB, all nine reference actions succeed, including two pending-order cancellations, the delivered watch return, and the refund calculation. The requested total $8,276.23 matches the listed arithmetic. This is a mixed cancel/return/refund-information workflow. |
| 22 | eligible_recommended_selected | all 7 succeeded; Name+ZIP authenticates against the pinned DB. All seven reference actions succeed. The scenario explicitly asks to move the default address, update the one eligible pending order, then restore the original profile address; the original street address is retrievable from the profile. The service must confirm each write. |
| 29 | eligible_recommended_selected | all 6 succeeded; Name+ZIP authentication succeeds. All six reference actions succeed, including two same-product delivered exchanges. The hidden item ID is a fact the customer is allowed to withhold while asking the agent to find the pending-order variant; this is not a false statement. The price comparison has explicit expected values. |
| 35 | eligible_recommended_selected | one nonfatal email lookup miss; alternate email and remaining actions succeeded; The known user ID and the second listed email identify the same account. The first email lookup is an expected nonfatal miss; the second succeeds, and all later reference actions succeed. Multiple personal email addresses are possible, and the scenario does not claim both are registered on the retail account. This provides a return plus pending-laptop-modification workflow. |
| 37 | eligible_recommended_selected | one nonfatal email lookup miss; name+ZIP and remaining actions succeeded; The first email lookup is not linked to a Retail account, but the supplied name+ZIP lookup succeeds and the subsequent order modification reference action succeeds. The task can be completed using the policy-approved authentication path and does not require claiming the email is the account email. It covers a payment/price constraint and a multi-item pending-order change. |
| 48 | exclude_recommended | all 6 actions succeeded; authentication adequacy assessed separately; The only explicit identity facts are a user ID and location; email is explicitly forgotten, and the full name is not separately stated. Retail policy requires email or full name+ZIP and forbids relying on user ID alone. Under the pinned user-simulation guideline, the customer cannot be assumed to invent or infer an unstated full name. The write reference action itself replays, but that does not resolve the authentication gap. |
| 52 | eligible_recommended_selected | all 5 succeeded; Name+ZIP authenticates. All five reference actions succeed. The delivered-camera exchange specifies the source item, same-product variant constraint, maximum zoom, and payment method; the reference action confirms an available same-product target. |
| 57 | exclude_recommended | 0 actions to replay; The task has no reference actions, no communicate_info, and no NL assertions, so there is no verifiable completion target. Its identity details also provide only user ID+location while the customer does not know email. This is not suitable for the activation panel. |
| 63 | exclude_recommended | all 7 reference actions succeeded; poem-line requirement assessed separately; The Retail modification itself has a valid reference sequence, but the scenario first requires the customer to provide a famous poem’s first line and specifies no poem or line. The pinned user-simulation guideline forbids inventing missing scenario facts. The out-of-domain request can be declined by Service, but the customer-side instruction as written cannot be faithfully enacted. |
| 72 | eligible_recommended_selected | both writes succeeded; Name+ZIP is provided. Both reference writes succeed. The scope and payment preference change before confirmation; the final gold action modifies only the backpack, while the pending-order address update is separate. This is a valid, nondeceptive preference revision when the agent confirms the final details before acting. |
| 73 | eligible_recommended_selected | write succeeded; Name+email are provided. The reference return succeeds for the four listed items and leaves the coffee machine out. The pinned Retail tool accepts the order, item set, and payment method. This candidate has already received a formal semantic review requirement in the plan; this AI review does not replace that sign-off. |
| 78 | eligible_recommended_selected | all 3 writes succeeded; Name+email authenticate. All three reference writes succeed on a pending order address, pending-order item modification, and a separate pending-order cancellation. Calling the item change an “exchange” colloquially is compatible with the pending-order modify tool; the reference action matches policy/tool semantics. |
| 81 | eligible_not_selected | both cancellations succeeded; Name+email authenticate and both whole-order cancellations succeed for pending orders with the stated “no longer needed” reason. Eligible, but the panel already includes multi-order cancellation in Task 16 and a separate cancellation branch in Task 78; this simpler cancellation-only task adds less workflow diversity than the selected mixed and conditional tasks. |
| 88 | eligible_recommended_selected | cancellation succeeded; Name+email authenticate and the expected whole-order cancellation succeeds. The scenario itself supplies “ordered by mistake” as the cancellation reason and does not say that this would be false; treat it as truthful unless the researcher knows otherwise. The conditional “modify if available, otherwise cancel” adds a distinct product-availability branch. Confirm this reason remains truthful during manual sign-off. |
| 103 | exclude_recommended | all 4 writes succeeded; task/action identifier mismatch remains; The reference writes replay successfully, but the task asks to return delivered goods and also says to tell the agent to “cancel” them, which can trigger a different policy action. The scenario does not specify how to resolve that conflict. In addition, the task is 103 while all reference action IDs are 104_* or unrelated, indicating a source-data binding concern. Exclude unless a researcher resolves both issues. |

## Researcher confirmation still needed

- Confirm the ten selected tasks and the eligibility judgments in the JSON. Task 35 and Task 37 each have an email lookup that does not match the Retail account; both remain resolvable through another supplied email or name+ZIP, but confirm that the task facts meet your consistency criterion.
- Confirm Task 88’s “ordered by mistake” cancellation reason is truthful for the customer.
- Confirm the pairwise diversity assessments, especially the address and exchange workflow pairs.
- After confirmation, create the normal `activation-smoke-review.json` using the existing task-semantic-review schema. Do not use this AI pre-review file as runner input.

## Excluded / not selected

- Task 48: I recommend excluding because the scenario provides only a user ID and location, not an explicit full name or email for the required authentication path.
- Task 57: no reference actions or scored communication assertions; no measurable completion target.
- Task 63: the prompt asks the user to provide an unspecified poem line, which the pinned simulator guidelines prohibit inventing.
- Task 103: all write calls replay, but “return” versus “cancel” is unresolved and its reference action IDs are bound to `104_*` rather than task `103`.
- Task 81 is eligible but not selected because it adds less diversity than the selected mixed and conditional workflows.
