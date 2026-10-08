# Official DeepSeek Flash: real Diagnoser preflight

User supplied a new official DeepSeek credential and clarified Flash, Evolver only.
Credential stored in a separate macOS Keychain generic password entry:
service api.deepseek.com/v1, account evotau-evolver-api-key. Secret supplied via
non-echoing stdin and saved using Security.framework, never in command-line args
or repository files. Probe reads it into its process-local OPENAI_API_KEY;
InferAI credentials and native runtime configuration are unchanged.

Official /v1/models returned deepseek-flash and deepseek-v4-pro. The exact Flash
ID was used, not an assumed deepseek-v4.1-pro alias. Endpoint:
https://api.deepseek.com/v1. No proxy; existing EvoTau generate/LiteLLM chat client
and unchanged V2 Diagnoser prompt/schema. No new provider adapter.

One request used the same ~30k-token compressed E20 Diagnoser context as prior
provider comparisons. Actual wire args: model deepseek-flash, thinking enabled,
reasoning_effort high, max_tokens65536, zero tools. Client timeout600s, no retries,
no temperature override. Cap1; no native episodes or full experiment launched.

**Success: HTTP200, finish_reason stop, full parseable JSON and strict diagnosis
contract accepted (2 clusters).** HTTP latency91.437s; total stage95.708s.
Provider usage: prompt31954, completion19883, reasoning19172 (a subset of completion,
not extra tokens to add again); total reported prompt+completion51837.
Visible content2950 characters; reasoning content83084 characters. Reasoning is
returned separately, not treated as the visible JSON. No empty visible output,
manual JSON repair or target/protected outcome label substitution was needed.

This validates official provider connectivity and the real Diagnoser contract.
It does not validate mutation quality, promotion, the remaining full V2 pipeline
or evolutionary effectiveness. Formal runs remain stopped; no model was silently
changed in a frozen run and no diagnosis was imported as a completed formal stage.
Model-generated causal claims remain hypotheses, even though the schema is valid.

Shared input is ../diagnoser-model-comparison-20261008/context.json; equality was
asserted. Per-call metadata, original parsed output, visible completion, usage
and wire logs are retained. The probe refuses to rerun if its report exists.
Ruff and whitespace checks passed. Official docs consulted:
https://api-docs.deepseek.com/guides/thinking_mode/
https://api-docs.deepseek.com/api/list-models/
