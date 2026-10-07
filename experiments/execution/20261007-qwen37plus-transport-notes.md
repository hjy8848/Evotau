# Qwen runtime transport conditions

The first four-worker attempt received 34 HTTP 200 and five HTTP 429 responses within about 28 seconds. The provider reported `group requests-per-minute limit exceeded`; its numerical allowance is unknown. Four simultaneous small-call probes had passed earlier, which did not establish sustained episode throughput.

The explicit resume retains four episode workers, original request payloads, zero automatic retries, unlimited total request budget, and the same frozen runtime/manifest. Only external transport scheduling changes: one thread-safe process-wide sliding window permits at most 30 InferAI chat dispatches per 60.1 seconds. This conservative rate is an operational setting, not a claim about the provider's exact limit. It also covers Pro calls using the same credential group. The runtime is not changed.

Original failed attempts remain in artifacts, excluded from fitness. Complete same-run episodes are reused if any. `actual-provider-http.jsonl` records per-call queue wait and transport scheduling separately; role API elapsed time includes queue wait, while wire response elapsed time excludes it. Budget remains `request_budget_cap: null` and there is no output-token budget newly imposed.
