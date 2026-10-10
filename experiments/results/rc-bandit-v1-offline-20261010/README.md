# RC-Bandit v1 offline evidence

All trajectories/reviewer support/rewards are **scripted**, not real τ-bench or API results.
No Keychain reads, real provider calls, native online episodes or H evaluation were started.

- `manifest.json`: implementation/config/source fingerprint and frozen E10 identity.
- `rc-scripted-result.json`: four observed classes, two synthetic allocation rounds, ten trials.
- `scripted-trajectories/`: forty complete synthetic source records (both endpoints × ten E IDs × two seeds).
- `rc-bandit-trials/`: immutable raw feedback/cost events; costs explicitly distinguish simulated units from zero real calls.
- `verification.json`: actual baseline/final test results, replay digest, defects corrected and remaining hypotheses.

The actual alternating G2 control flow is tested separately with Fake Providers/Runner in
`tests/test_repair_conditioned_bandit.py`; this allocation fixture is not an effectiveness experiment.
The first round covers all five arms; the second uses the two scripted discoveries, deduplicates
repeated findings and maintains open exploration. These priorities cannot support a research-benefit claim.

See `docs/research/repair-conditioned-bandit-v1.md` for frozen protocol, recovery and future
E-only cost-matched study.
