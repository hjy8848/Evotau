# Retail E20/G5/P1 — checkpoint continuation

Status: **failed**. Completed generations: **1/5**. No V/H evaluation.

| Gen | Customer before | Customer after | S before acc | challenged acc | repaired acc | Service accepted |
|---|---|---|---:|---:|---:|---|
| 0 | 63be7e41e020bc00 | 4f0efed3a149e457 | 95% | 50% | 65% | True (add) |
| 1 | N/A | N/A | N/A | N/A | N/A | not completed |
| 2 | N/A | N/A | N/A | N/A | N/A | not completed |
| 3 | N/A | N/A | N/A | N/A | N/A | not completed |
| 4 | N/A | N/A | N/A | N/A | N/A | not completed |

## Continuation conditions

Gen0 and 60 complete native episodes were imported byte-for-byte from `evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g2-p1-formal-20261006`. Gen0 achieved 95%→50%→65%, accepted one ADD skill. Gen1 onward increases Evolver single-request output allowance to `max_tokens=65536`; original run sent no explicit output maximum and stopped at 8192 with length-truncated JSON. This is an explicit continuation with changed output arguments, **not a fresh five-generation replicate or five generations under identical Evolver output settings**.

Target experiment: `evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g5-p1-continuation-20261007`. Source commit: `a8ba7daae81ebd164d20b6e785f01b5b235355c2`. Source SHA256: `965497fcfd0260fdf20cc2525fecc024d8853bd82d1baa44730f1b0ce3902f3e`. Manifest SHA256: `3d3d9ea7e62c647a74e479b201b5bad7089b78446db4d26bf9d4c58e9a3a1eba`. Config SHA256: `1e32709cc1612fc2fc1cfa62d4e9b22eab0c9f1bbf9a3ee4cead4294cdbafb76`.

E20 IDs: `66 92 29 67 106 22 69 98 93 88 4 21 8 54 107 48 52 80 35 16`. Fixed seed1, candidate1, serial episodes, max_steps32, unlimited request budget, V/H disabled. Runtime Flash roles/args, Pro model/thinking/high-effort request, prompts, frozen source, policy/tools/native scoring and selection remain unchanged. Actual wire parameters are recorded; provider compliance with named high effort remains unverified.

`continuation-provenance.json` retains parent usage/timing and original artifact hashes; `continuation-origin/` retains original manifest/checkpoint/state/context. Only the cloned checkpoint manifest binding and panel-reference manifest binding were changed for the new continuation; original Gen0 documents preserve parent provenance. Original run remains intact.

## API / time

| Role | Total calls (incl. parent) | New calls | Prompt tokens | Completion tokens | API seconds | Mean seconds |
|---|---:|---:|---:|---:|---:|---:|
| customer | 467 | 25 | 601,782 | 17,369 | 1550.41 | 3.32 |
| service | 770 | 35 | 5,698,902 | 134,582 | 3755.82 | 4.88 |
| evaluator | 20 | 1 | 123,765 | 1,781 | 101.10 | 5.05 |
| customer_evolver | 3 | 1 | 397,995 | 21,725 | 250.97 | 83.66 |
| service_evolver | 1 | 0 | 136,752 | 5,271 | 70.71 | 70.71 |
| reviewer | 0 | 0 | 0 | 0 | 0.00 | 0.00 |
| customer_judge | 0 | 0 | 0 | 0 | 0.00 | 0.00 |
| service_judge | 0 | 0 | 0 | 0 | 0.00 | 0.00 |

Including parent: 1,261 API calls, 6,959,196 prompt tokens, 180,728 completion tokens. New continuation: 62 calls, 461,995 prompt tokens, 21,096 completion tokens.

Continuation wall-clock: 370.35s. Including parent active runtime: 5752.43s (excludes pause gaps). Unique complete episodes: 62; panel references: 82; reused references: 20; incomplete attempts: 1. Peak observed provider concurrency: 1.

Flash actual thinking disabled 1257/1257; reasoning tokens 0; empty successful responses 0. Pro request/response metadata and role/call-name usage are fully recorded in the JSON summary and wire log.

## Per-generation proposals and paired outcomes

### Gen0

Incumbent E accuracy 95%; candidate 50%; selected 50%, selected Customer=0. Incumbent cache reuse 0/20, newly run 20.

Customer strategy:

In a single opening message, request a large multi-part account cleanup: cancel every order that is still cancellable, return several delivered items while explicitly keeping one favored item untouched, and add a conditional instruction to exchange that favored item only if a cheaper replacement with the same key attributes exists. Withhold your decision on the conditional exchange until the agent enumerates the available options, and do not ask for the final aggregate amount (e.g., total refund) until every action has been executed. Keep replies terse so the agent must plan, confirm, and carry out the whole workflow without relying on you to volunteer additional details.

Customer paired outcomes:

- pass_to_fail: 10; `92, 67, 98, 93, 4, 21, 8, 52, 80, 35`
- fail_to_pass: 1; `54`
- pass_to_pass: 9; `66, 29, 106, 22, 69, 88, 107, 48, 16`
- fail_to_fail: 0; `none`

Service operation add, accepted True. Old E accuracy 0.5; candidate E accuracy 0.65. V disabled.

Service analysis:

The empty skill memory provides no coverage for the recurring multi-part account cleanup pattern. Two independent failed episodes (tasks 93 and 98) hit the exact same backend error: the agent processed item returns on a delivered order (order status became 'return requested') and then attempted to exchange a separate kept item from the same order, receiving "Error: Non-delivered order cannot be exchanged". This is a deterministic procedural failure, not stochastic: the exchange tool requires the order to still be in 'delivered' status, and any return changes the order status. The evidence also shows a repeated conditional-exchange pattern where the customer withholds the exchange decision until the agent enumerates available options; in both failed episodes the agent finalized returns before eliciting/enacting the exchange. A small, reusable skill is warranted to sequence delivered-order actions correctly and defer status-changing actions until exchange decisions are complete.

Service paired outcomes:

- pass_to_fail: 1; `54`
- fail_to_pass: 4; `92, 4, 21, 35`
- pass_to_pass: 9; `66, 29, 106, 22, 69, 88, 107, 48, 16`
- fail_to_fail: 6; `67, 98, 93, 8, 52, 80`

Memory before:

```json
{
  "role": "service",
  "carrier": "skill_memory_v1",
  "strategy_id": "0b51396195500388",
  "strategy": {
    "skills": []
  },
  "skill_count": 0,
  "rendered_skill_chars": 0,
  "rendered_skill_tokens": null
}
```

Memory after:

```json
{
  "role": "service",
  "carrier": "skill_memory_v1",
  "strategy_id": "28940cc308f6a614",
  "strategy": {
    "skills": [
      {
        "skill_id": "skill-0001",
        "trigger": "When a customer requests multiple order actions in one conversation, especially involving a delivered order where some items are to be returned while another item is kept and may be exchanged (often conditionally, e.g., 'only if a cheaper replacement with the same key features exists — I will decide once you list the options').",
        "guidance": "Before making any tool call, build a complete per-order plan covering which items will be returned, exchanged, kept, or cancelled, and obtain explicit confirmation on the full plan. Exchange tools only work while an order's status is still 'delivered': processing a return for any item in an order immediately changes that order's status (e.g., to 'return requested'), after which the backend rejects exchanges for the remaining items. Therefore, if a delivered order has both items to exchange and items to return, resolve the exchange before processing any return on that order. If the customer is deciding conditionally, enumerate the matching available variants of the same product (variants that match the requested key attributes, and only the cheaper ones when a cheaper option was the stated condition) BEFORE processing any returns on that order, then let the customer decide and only then proceed. Execute all exchange items of an order in a single exchange call, since the exchange tool can be called only once per order. Do not partially return items from an order whose remaining items still require an exchange decision."
      }
    ]
  },
  "skill_count": 1,
  "rendered_skill_chars": 2126,
  "rendered_skill_tokens": null
}
```

## Last accepted active memory and provenance

```json
{
  "service": {
    "skills": [
      {
        "skill_id": "skill-0001",
        "trigger": "When a customer requests multiple order actions in one conversation, especially involving a delivered order where some items are to be returned while another item is kept and may be exchanged (often conditionally, e.g., 'only if a cheaper replacement with the same key features exists — I will decide once you list the options').",
        "guidance": "Before making any tool call, build a complete per-order plan covering which items will be returned, exchanged, kept, or cancelled, and obtain explicit confirmation on the full plan. Exchange tools only work while an order's status is still 'delivered': processing a return for any item in an order immediately changes that order's status (e.g., to 'return requested'), after which the backend rejects exchanges for the remaining items. Therefore, if a delivered order has both items to exchange and items to return, resolve the exchange before processing any return on that order. If the customer is deciding conditionally, enumerate the matching available variants of the same product (variants that match the requested key attributes, and only the cheaper ones when a cheaper option was the stated condition) BEFORE processing any returns on that order, then let the customer decide and only then proceed. Execute all exchange items of an order in a single exchange call, since the exchange tool can be called only once per order. Do not partially return items from an order whose remaining items still require an exchange decision."
      }
    ]
  },
  "provenance": [
    {
      "skill_id": "skill-0001",
      "created_generation": 0,
      "updated_generation": 0,
      "source_task_ids": [
        "66",
        "92",
        "29",
        "67",
        "106",
        "22",
        "69",
        "98",
        "93",
        "88",
        "4",
        "21",
        "8",
        "54",
        "107",
        "48",
        "52",
        "80",
        "35",
        "16"
      ]
    }
  ],
  "skill_count_trajectory": [
    0,
    1
  ]
}
```

## Failure / validity / resume

```json
{
  "failure": {
    "stage": "evolution",
    "failure_type": "NativeEpisodeRunError",
    "failure_message": "JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
    "task_id": "29",
    "panel_name": "generation-1-customer-candidate-0",
    "call_name": null,
    "diagnostics_ref": "episodes/85ff610fe07e4cb99adcf88d0b00fdf1/incomplete-run.json",
    "last_generation_stage": {
      "generation": 1,
      "stage": "customer_proposals_ready"
    }
  },
  "incomplete_attempts": 1,
  "integrity": {
    "wire_requests": 1261,
    "wire_responses": 1261,
    "provider_log_calls": 1261,
    "provider_budget_calls": 1261,
    "wire_budget_and_log_reconciled": true,
    "successful_response_ids_unique": true,
    "all_provider_retries_zero": true,
    "all_requests_settled": true,
    "budget_inflight_and_reserved_zero": true,
    "panel_order_checks": {
      "generation-0": true
    }
  }
}
```

Failures remain fail-closed. No automatic provider retries, model changes, JSON repair or skipping tasks. The original failed Gen1 Evolver call remains in inherited diagnostics and is distinguished from continuation failures by call ID and lifecycle state. Cache/checkpoint remain available for explicit resume. All tables use complete panels; no unknown result is counted as failure. Cross-Customer accuracy is not a same-condition Service improvement. V/H disabled, Retail only and seed1: no generalization claim.

Full native conversations, tools and evaluation are in `episodes/`; generation and SkillMemory artifacts, `actual-provider-http.jsonl`, checkpoint snapshot, provider logs, summary and SHA256 inventory are archived. Execution helpers only supervise/archive the existing runner; no algorithm/source edits and no extra model calls.
