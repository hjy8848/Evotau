# Final offline V2 native integration evidence

This is an engineering smoke, not a live InferAI run or research result.

- Pinned τ-bench: real UserSimulator, LLMAgent, Retail tools/backend and native evaluator.
- Completion boundary: deterministic local scripted ModelResponse, **zero network provider calls**.
- E: task 66 only; G1; max_steps=2; serial; V/H disabled and H sealed.
- Five complete native simulations; 17 locally instrumented generate calls.
- Explicit resume: zero additional calls, exact usage retained.
- Customer candidate ties incumbent; Service ADD candidate has no native fix and is correctly rejected by screen. No promoted skill and no improvement claim.
- Candidate execution still exercises per-turn selection/injection and saves activation traces. It does not validate a real activator's reasoning.

`offline-evidence.json` distinguishes network calls from instrumented local calls. The manifest retains provider-enabled dispatch instrumentation so the normal production runner can be exercised; its model IDs are explicitly `offline-*` and the test replaces completion before dispatch. The instrumentation counts must never be mistaken for paid/live requests.

Contents preserve the original config, immutable manifest/stages/call records, checkpoint, native simulations/rewards and decisions. Runtime source/config fingerprints describe the tested working tree. Opening it does not launch a provider request:

```bash
evotau-web --project-root experiments/results/skill-evolution-v2-offline-native-20261007 --port 8001
```

Reproduce into a **new** directory:

```bash
python experiments/execution/run-v2-offline-smoke.py \
  --tau2-data-dir /path/to/pinned/data \
  --export /tmp/new-v2-native-evidence
```

The deterministic fake-runner tests independently verify three candidates, explicit bad/regressive proposals, locally useful archive entries, complementary crossover, promotion, two generations and interruption recovery. Those synthetic outcomes are not empirical τ-bench accuracy gains.

Final verification includes correct shared request accounting: all 17 local calls are counted by provider usage and role usage; resume adds zero calls. Runtime/docs commit: `fa9f391`. Full offline suite: 179 passed. See `verification.json`.
