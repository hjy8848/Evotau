# Retail E20/G2/P1 — Gen0 completed, Gen1 Evolver output truncated

**第一轮（Gen0）完整完成，有 challenge → repair 信号。两轮实验尚未完成；Gen1 Customer Evolver JSON截断，原始状态保持failed，实验暂停。**

| Gen | Customer before | Customer after | S before acc | challenged acc | repaired acc | Service accepted |
|---|---|---|---:|---:|---:|---|
| 0 | C0 | C1 (candidate 0) | 95% (19/20) | 50% (10/20) | 65% (13/20) | true, ADD |
| 1 | C1 | 未产生合法candidate | 65% (13/20, cache reuse) | N/A | N/A | 未评估 |

Customer incumbent E accuracy=95%；唯一candidate=50%；selected=50%。Customer选择严格argmin，Service E-only acceptance严格higher accuracy。Service更新确实发生：SkillMemory 0→1、S0→S1；65%是同一挑战下的修复结果，不是native C0下的S1 accuracy。V/H关闭，无泛化结果，未加载held-out task content。

## Frozen configuration and provenance

- Experiment: `evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g2-p1-formal-20261006`
- Source commit at execution: `2fd5d7839b178fb80a9f33c003461e6548fa00e9`
- Source SHA256: `965497fcfd0260fdf20cc2525fecc024d8853bd82d1baa44730f1b0ce3902f3e`
- Manifest SHA256: `d44ece1d268afc42cbc04169a3fb321e01bf5a9a07eaaa8e06177cea70670995`
- Config SHA256: `0bf7e68ec93f5e5e20e59546a9713508552aca39c42c8574e149bf13229b8038`
- E20 train IDs: `66 92 29 67 106 22 69 98 93 88 4 21 8 54 107 48 52 80 35 16`；与V/H/excluded互斥。E顺序固定。
- Requested G2/C1/P1，seed=fitness seed=1，max_steps32，request cap=null；V/H flags=false。
- Agent/Customer/Evaluator=`openai/deepseek-v4-flash`，temperature0，InferAI，thinking disabled；Evolvers=`openai/deepseek-v4-pro`，thinking enabled，配置请求reasoning_effort=high，无temperature。
- Actual Flash wire args: `[{"model": "deepseek-v4-flash", "thinking": {"type": "disabled"}, "temperature": 0.0}]`
- Actual Pro wire args: `[{"model": "deepseek-v4-pro", "thinking": {"type": "enabled"}}]`。未在实际HTTP body观察到named reasoning_effort字段，high compliance未独立验证。
- 源码、prompts、benchmark semantics、selection/executor均未改变。114 tests、lint及diff检查在启动前通过。P1使用独立新config/run/checkpoint，未复用P2/P4或旧实验结果。

## Paired task changes (fixed E, seed 1)

Customer C0×S0 → C1×S0:

- pass_to_fail: 10; IDs `92, 67, 98, 93, 4, 21, 8, 52, 80, 35`
- fail_to_pass: 1; IDs `54`
- pass_to_pass: 9; IDs `66, 29, 106, 22, 69, 88, 107, 48, 16`
- fail_to_fail: 0; IDs `none`

Service C1×S0 → C1×S1:

- pass_to_fail: 1; IDs `54`
- fail_to_pass: 4; IDs `92, 4, 21, 35`
- pass_to_pass: 9; IDs `66, 29, 106, 22, 69, 88, 107, 48, 16`
- fail_to_fail: 6; IDs `67, 98, 93, 8, 52, 80`

Service修复4个task（92/4/21/35），退步1个（54），净+3/20=+15个百分点；未完全恢复到95%。所有paired transitions均来自完整native结果，unknown未作为failure或fitness。

## Actual Customer mutation

In a single opening message, request a large multi-part account cleanup: cancel every order that is still cancellable, return several delivered items while explicitly keeping one favored item untouched, and add a conditional instruction to exchange that favored item only if a cheaper replacement with the same key attributes exists. Withhold your decision on the conditional exchange until the agent enumerates the available options, and do not ask for the final aggregate amount (e.g., total refund) until every action has been executed. Keep replies terse so the agent must plan, confirm, and carry out the whole workflow without relying on you to volunteer additional details.

## Actual accepted Service skill

Operation: ADD. ID: `skill-0001`. Active strategy ID: `28940cc308f6a614`.

Service Evolver analysis (model interpretation):

The empty skill memory provides no coverage for the recurring multi-part account cleanup pattern. Two independent failed episodes (tasks 93 and 98) hit the exact same backend error: the agent processed item returns on a delivered order (order status became 'return requested') and then attempted to exchange a separate kept item from the same order, receiving "Error: Non-delivered order cannot be exchanged". This is a deterministic procedural failure, not stochastic: the exchange tool requires the order to still be in 'delivered' status, and any return changes the order status. The evidence also shows a repeated conditional-exchange pattern where the customer withholds the exchange decision until the agent enumerates available options; in both failed episodes the agent finalized returns before eliciting/enacting the exchange. A small, reusable skill is warranted to sequence delivered-order actions correctly and defer status-changing actions until exchange decisions are complete.

Trigger:

When a customer requests multiple order actions in one conversation, especially involving a delivered order where some items are to be returned while another item is kept and may be exchanged (often conditionally, e.g., 'only if a cheaper replacement with the same key features exists — I will decide once you list the options').

Guidance:

Before making any tool call, build a complete per-order plan covering which items will be returned, exchanged, kept, or cancelled, and obtain explicit confirmation on the full plan. Exchange tools only work while an order's status is still 'delivered': processing a return for any item in an order immediately changes that order's status (e.g., to 'return requested'), after which the backend rejects exchanges for the remaining items. Therefore, if a delivered order has both items to exchange and items to return, resolve the exchange before processing any return on that order. If the customer is deciding conditionally, enumerate the matching available variants of the same product (variants that match the requested key attributes, and only the cheaper ones when a cheaper option was the stated condition) BEFORE processing any returns on that order, then let the customer decide and only then proceed. Execute all exchange items of an order in a single exchange call, since the exchange tool can be called only once per order. Do not partially return items from an order whose remaining items still require an exchange decision.

技能核心是先完成换货决策/执行，再发起会改变订单状态的退货。Evolver引用task93/98中的backend错误作为提案依据；这两个task在candidate replay仍未通过，不能据此宣称它们已被修好。实际修复/退步ID以上面的paired结果为准。完整ADD提案、来源task、输入hash、memory和provenance保存在generation/service-memory/checkpoint文件中。

## Gen1 failure and checkpoint

Gen0 checkpoint completed_generation=0；Gen1 incumbent复用Gen0 C1×S1的全部20个完整条件。共60份unique complete episodes，80个panel references（20 reuse），无incomplete native episode，无重复运行成功条件。

Gen1 Customer Evolver第三次Pro调用HTTP200、finish_reason=length、completion_tokens=8192，其中reasoning_tokens=8065，visible content仅623字符，JSON字符串未闭合。实际request没有max_tokens/max_completion_tokens，也没有temperature；不是显式EvoTau输出token上限。8192截断来源的具体provider/default配置尚未证实。严格parser报`EvolverJSONError`，没有修补JSON、采纳半截candidate、跳task或继续Service selection。

- Failure ref: `evolver-calls/eebfe7913c394cd0974005e07d8953a2/failure.json`
- Last stage: generation1/customer_incumbent_complete
- No 429/504/transport failures: 1199/1199 responses HTTP200；其中1个HTTP成功回复在Evolver JSON语义解析时失败，API success count不代表evolution完成。
- Flash: 1196/1196 actual thinking disabled，reasoning_tokens=0，reasoning text=0，异常空completion=0。
- Pro: 3 calls，actual thinking enabled，reasoning tokens总计13997；1次length截断。
- No retries/resume/model substitution. Native telemetry和wire所见峰值均1。Provider requests已全部settled，reserved/inflight=0。
- 没有完整alternating-result.json或Gen1 result；不把此次快照标为完整G2/正式泛化实验。

## API and time (includes failed Gen1 Evolver call)

| Role | Calls | Prompt tokens | Completion tokens | API elapsed seconds | Average seconds |
|---|---:|---:|---:|---:|---:|
| customer | 442 | 558,441 | 16,421 | 1476.22 | 3.34 |
| service | 735 | 5,428,500 | 126,854 | 3576.98 | 4.87 |
| evaluator | 19 | 113,814 | 1,624 | 94.50 | 4.97 |
| customer_evolver | 2 | 259,694 | 9,462 | 143.26 | 71.63 |
| service_evolver | 1 | 136,752 | 5,271 | 70.71 | 70.71 |
| reviewer | 0 | 0 | 0 | 0.00 | 0.00 |
| customer_judge | 0 | 0 | 0 | 0.00 | 0.00 |
| service_judge | 0 | 0 | 0 | 0.00 | 0.00 |

Total: **1,199 calls；6,497,201 prompt tokens；159,632 completion tokens；6,656,833 total tokens；wall-clock 5382.08秒（约89.7分钟）**。Gen0 wall-clock 5269.09秒；Customer phase 3452.32秒；Service phase 1816.76秒。V/H未运行，耗时0。Reviewer/Customer Judge/Service Judge calls均0。

## Files and future resume

- `formal-attempt-summary.json`: accuracy、paired IDs、roles/call names统计、wire验证、timing、最后已接受memory/provenance。
- `episodes/`: 60场完整Customer/Service对话、工具结果、native evaluation和telemetry。
- `generation-0000.json` / proposal files / `service-memory/`: 完整第一轮演化证据。
- `evolver-calls/` / `evolver-failures.json`: 第二轮失败input/output/parse证据。
- `checkpoint-snapshot.json`: 只读归档的Gen0恢复点；原始run/checkpoint在本地保留。
- `actual-provider-http.jsonl`: 去除凭据/headers/完整messages/reasoning文本的实际wire metadata。
- `config.yaml` / `manifest.json` / `preflight.json` / `run-execution-state.json`: 冻结条件、offline验证和真实failed状态。
- `archive-sha256.json`: 归档文件校验清单。

用户暂停后仅做结果归档，未发送额外provider请求。后续明确resume可复用已有60场完整条件和Gen0 checkpoint；JSON截断问题仍需在恢复时处理。改变源码或请求args应明确新条件/manifest，不能偷偷改变本次冻结实验。
