# Airline Analyst recovery v4 — exploratory G2

Source commit: `be651f3e8de8b48ff64554ebe3018b471ed4f99c`. Completed 2026-10-10 13:51 Asia/Shanghai. Runtime: gateway Qwen3.7-plus, thinking disabled; Evolver: official DeepSeek Flash, thinking enabled. E10, V20, H20, two generations, two episode workers; E/Screen/H seed1, V seeds1,2.

- Gen0: E 8/10; Customer semantic rejection; Analyst schema/evidence validation exhausted (original plus two recoveries); no Skill candidate.
- Gen1: E 8/10; Customer semantic rejection; two Skill proposals: one semantic rejection, one Screen rejection (0 fixes, 0 regressions; 5/6 before and after). No V evaluation and no promotion. Final SkillMemory empty; S0 equals ST.
- Native H: 15/20 (75%); identical ST scores reused. Fresh Customer rejected, fresh-adaptive H unavailable. H has one seed; Pass4 unavailable.
- 36 completed episodes; 732 successful provider calls; 4,622,772 prompt tokens and 262,393 completion tokens. No transport failures. Cumulative execution 2,321.501510 seconds (38m42s), excluding downtime between invocations.
- One explicit audited Customer schema replacement after the model returned three candidates rather than one. Original output and failure retained. Controller recovery: five logical calls, three originally valid, one recovered, one exhausted; three extra requests. Recovery proves workflow progress, not mutation effectiveness.

A/A remains uncalibrated; H20 was previously used in historical experiments. This is exploratory evidence, not final generalization proof. H summary termination metadata is incomplete (`unknown`); consult original trajectories rather than interpreting summary stuck counts as validated.

Top-level JSON reports and frozen config are readable directly. `full-artifacts.tar.gz` includes every original run file, all native trajectories, provider/evolver responses, immutable journals, recovery evidence and logs. `artifact-inventory.json` lists hashes; every archive member was verified. Extract into an independent directory, never into a live run:

```sh
mkdir extracted
tar -xzf full-artifacts.tar.gz -C extracted
```
