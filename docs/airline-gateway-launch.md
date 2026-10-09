# Airline gateway launch

The native Agent, Customer, Evaluator and Activator use
`openai/dashscope/qwen3.7-plus` at `http://10.130.138.46:8010/v1`.
Their frozen `enable_thinking: false` is translated into
`extra_body.enable_thinking=false`. The InferAI `thinking.type` parameter did not
turn off reasoning at this gateway and must not be reused. Official DeepSeek
`openai/deepseek-flash` remains the Evolver, with its existing model settings.
Credentials are read from Keychain and scoped to the exact host/model; they are
never included in artifacts. No automatic provider fallback or retry is added.

`configs/airline-gateway-qwen37plus-aa-p2.yaml` is an independent E10 × four
seeds × two repetitions A/A diagnostic (80 episodes, cap 20,000 requests).
The two repetitions have independent cache namespaces. Within each repetition,
at most two independent episodes run concurrently; turns remain sequential.
Errors stop new dispatch, allow already running episodes to finish and preserve
completed artifacts. Resume uses the same config/output and cumulative budget.

```sh
PYTHONPATH=src TAU2_DATA_DIR=/Users/spring/.cache/evotau/tau2-data-b7ea9074 \
/Users/spring/RSI/Evotau/.venv/bin/python experiments/execution/run-airline-gateway.py \
  --config configs/airline-gateway-qwen37plus-aa-p2.yaml \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074 \
  --mode aa --approved-request-cap 20000 --execute \
  --stop-before-next-episode-file /tmp/evotau-airline-gateway.pause
```

Omit `--execute` for offline validation. To pause, create the specified pause file;
remove it and repeat the same command to resume. At most two episodes already
running may finish. Failures remain in `operator-failure.json`,
`diagnostic-failure.json` and sanitized `actual-provider-http.jsonl`; their
presence is historical evidence, not necessarily the status of a resumed run.

`configs/airline-gateway-qwen37plus-formal-p2.yaml` is a **pending template** for
E10/V20/H20, G2, max_steps200, concurrency2, request cap200,000. Airline has only
30 train tasks, so E20/V20 is impossible. Formal execution remains blocked until
Airline A/A calibration is actually completed, thresholds are reviewed/frozen,
and the finite formal budget is explicitly confirmed. Do not set
`calibration_confirmed=true` merely to bypass this check. A/A and formal runs use
separate identities and no old InferAI/Retail scores are imported. H is not
loaded during A/A or launch validation.

The native probe validated two concurrent requests and valid JSON/tool arguments;
it did not establish sustained throughput, native episode completion, statistical
calibration or research effectiveness. Pricing metadata for the gateway model
is not available in the local LiteLLM cost table; report observed request/token
usage rather than claiming an exact monetary price.

## Explicit direct launch (2026-10-09)

After being told calibration is incomplete and the proposed formal cap is 200,000,
the user explicitly requested direct launch. The independent frozen config
`configs/airline-gateway-qwen37plus-direct-launch-p2.yaml` records that authorization
with `allow_uncalibrated_launch: true`. `calibration_confirmed` remains false.
No statistical, regression, policy or data-isolation threshold is relaxed.
This is an uncalibrated, predeclared experiment; do not label it A/A calibrated.
The default remains to require calibration for other V-primary configurations.
Run with the gateway launcher, `--mode formal --approved-request-cap 200000 --execute`.
