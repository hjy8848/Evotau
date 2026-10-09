# User experiment preferences

2026-10-09: The user explicitly instructed: “别设置各种上限了除非我叫你”.
Do not independently introduce API spending/request caps, Evolver output token
caps, or evidence/context length stopping caps for this user's EvoTau experiments.
Use nullable/unbounded settings where the provider supports them; disclose any
provider-imposed limits rather than claiming they were removed. Preserve explicitly
requested experimental conditions (currently concurrency2, native max_steps200,
G2 and declared candidate search strength), scientific validation/data boundaries,
and fail-closed handling of real provider/runtime errors. Future limited-budget
requests override this preference for their explicitly stated scope.
