# Final Open-RC offline verification — Synthetic only

Implementation: `c006fc2f56ba90345be583874c31bebd930c9539`.
Baseline: `6fc0df82d6716f328be89bb21895e5325faffc9f`.

- Full pytest: **503 passed**, 6 dependency warnings, 42.34 seconds.
- New protocol tests: 34; Ruff and git diff checks passed.
- Four independent configurations dry validated; no V/H task content loaded by dry validation.
- Real orchestration exercised with explicitly Fake Providers/Runner: toy E3/V1, G2.
  An unselected Customer produced a supported synthetic Discovery, which reached Service,
  passed targeted native-shaped checks, E protection and V preservation, and was archived.
- 85 original Synthetic trajectory files retained; 24 repair-review original-message
  references were matched to their saved projected-message hashes.
- CLI replay completed successfully with the identical synthetic-result SHA-256:
  `9e5a6dbfad0a954b1c375fe8d359c127aeae9c8ecfe2c7db384ff3df6302aa52`.
- Tests also cover rejection, unknown outcomes, invalid evidence, interrupted recovery,
  no duplicate requests/credit, no fixed-arm use and legacy Bandit behavior.

**Zero real provider calls, zero online native episodes, no Keychain access, no H loading.**
The Airline E10 template was only checked against pinned local inputs. The toy G2
fixture does not report Airline benchmark accuracy, real novelty, causal Skill gains,
statistical calibration or generalization. No remote push was performed.

See `verification.json`, `pytest.log`, `dry-validations.json`, the frozen
`parent-template-manifest.json` and `synthetic-g2-E3-V1/synthetic-manifest.json`.
Full Trial indexes, source trajectories, generation/stage artifacts, target decisions,
archive use events and checkpoint are retained under the Synthetic directory.
