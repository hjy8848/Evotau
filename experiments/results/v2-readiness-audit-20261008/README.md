# V2 readiness audit — NO-GO

No formal E/V/H experiment started. See the
[audit and operational runbook](../../../research/audits/2026-10-08-v2-readiness-audit.md)
and [metadata-only statistical/budget plan](../../../research/audits/2026-10-08-v2-readiness-plan.json).

Baseline `cae257557e302e843b0d29de1018badd058f1ed0`; E3/E5 runtime frozen at
`ea99eebb6a049ea8c5050f79981812923987c46b`, source hash
`eaee4c0de210cc1e2317d8cfaa9d886f7c05e9ad0a9b94ddc70093c83b6b2cd8`.
Readiness helpers/tests/docs were subsequently added without changing that
runtime hash. Representative probes preceded the frozen implementation commit;
their unchanged plans/contexts/wire artifacts are evidence of the actual requests,
not a bitwise source-replay certificate.

| Evidence | Result |
|---|---|
| Engineering tests | 226 passed |
| Four non-stream representative Pro calls | Diagnoser 504; other three stages complete legal schemas |
| Four streaming representative Pro calls | Diagnoser and Mutator 504; two Validators complete legal schemas |
| E3/V3/H-off G1 rehearsal | 3/3 native success; Customer rejected; no diagnosis cluster; no screen/gate |
| Independent E5/V3/H-off G1 rehearsal | 4/5 native success; Customer rejected; Diagnoser 504 at 120.14s; fail-closed |
| E3 completed replay | Zero new requests; 60 immutable files unchanged |
| E5 after-error initialization | Five cached conditions/four frozen stages validated; zero requests; budget restored |

Required **diagnosis → mutation → screen → gate** live coverage remains incomplete.
A completed empty-repair generation is not counted as that proof. No skill was
accepted, no fitness was fabricated, no model was switched, no failed stage was
automatically retried. All H content remains unsealed by these runs; V panels were
configured but not scored because repair/screen stages were not reached.

[`summary.json`](summary.json) counts all live audit usage without double-counting
E3 calls carried to E5: **128 requests**, **784,330 reported prompt tokens**,
**66,684 reported completion tokens**, four requests with unavailable usage.
Six unique completed native episodes were executed; two copied baseline records
are not new episodes. Reviewer/Customer Judge/Service Judge each issued zero calls.
The proposed formal cumulative cap is 150,000 calls and remains unconfirmed;
the formal pilot draft stays disabled.

The two representative directories include exact source-context hashes, sanitized
wire metadata and stage outcomes. The rehearsal directories contain native
simulations/scores, typed output or failure evidence, immutable journals, budget,
manifest and recovery verification. E3 has a generation checkpoint; E5 does not
because its generation failed before commit. E5's explicit baseline-import record
contains original/derived hashes and all prior cumulative usage. Only activation
manifest-binding metadata was transformed; native trajectories and scores were
copied unchanged. No Customer/search stages crossed the expanded E panel.

[`sha256-index.json`](sha256-index.json) covers every archived evidence file.
The native provider IDs have no verified installed price mapping: do not interpret
any native cost-zero placeholder as free inference. Failed provider-call token
usage is unknown, not zero.
