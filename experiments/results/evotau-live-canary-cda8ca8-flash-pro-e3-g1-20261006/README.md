# Live release canary · cda8ca8

**本轮 canary PASS：真实小规模完整路径正常结束。没有启动正式 E20×2。**

运行源码为 `cda8ca81e3c2155b7752552762ce7969789c634a`，source SHA256 `965497fcfd0260fdf20cc2525fecc024d8853bd82d1baa44730f1b0ce3902f3e`。本轮没有修改 `src/`、模型 adapter、prompts 或 selection。额外的进程内 HTTP observer 只记录实际 request 参数和 response metadata，原请求/回复不修改，不 mock provider；没有记录 Authorization/key、完整 request messages 或 reasoning 文本。

## 配置与结果

E = `66, 92, 29`（原 panel 的前 3 个合法 train tasks；不按表现选择），G=1，Customer candidates=1，fitness seed=1，max_steps=32，episode concurrency=1，request_budget_cap=null。V/H关闭。Flash负责Customer/Service/native evaluator；Pro负责两个Evolver，SkillMemory从空开始。

| Gen | Incumbent E | Candidate E | Selected E | Service operation | Candidate replay | Service accepted |
|---|---:|---:|---:|---|---|---|
| 0 | 2/3 (66.7%) | 3/3 (100%) | 2/3 (66.7%) | NO_OP | 按规则跳过 | false |

Customer candidate 未降低 accuracy，正确保留 incumbent。Service Evolver认为当前可见失败证据不足以支持可复用技能，返回NO_OP；空 SkillMemory 保持不变，checkpoint提交 completed_generation=0。**没有发生 Service 更新，也不是有效进化/泛化证据。**本轮没有覆盖 live ADD/UPDATE rollout分支；NO_OP/skip是需求允许的完整路径。

## 三项关键检查

1. **真实 Flash HTTP body：127/127 为 `thinking: {"type":"disabled"}`，temperature=0.0。**不仅是converter返回值。
2. **127/127 Flash回复 reasoning_tokens=0；没有reasoning文本，也没有无内容且无tool-call的空输出。**129个总请求全部HTTP200且有可用输出；Pro两个回复均开启thinking、返回可见JSON，reasoning tokens共13,397。Service仅tool-call而没有文本属于合法输出。
3. **6/6 episode完整并有有效 boolean native score，两个E panels齐全且顺序固定；没有unknown计失败、跳task、provider失败或隐式重试。**9项额外离线负例回归通过，验证无效reward/未知fitness、failed/paused收尾、cache/reservation复用；此次没有真实provider失败，不能声称实测了live故障恢复。

原生tools/backend共保存49个tool结果；task29执行两次native NL assertion LLM evaluation。初始task29的reward=0是合法native task failure，不是缺失评分。Customer/Service phase严格顺序，H/V任务内容未载入，Service input不含隐藏user_scenario；场景原文hash保持不变。Reviewer/Customer Judge/Service Judge calls均为0。

## API与耗时

| Role | Calls | Prompt tokens | Completion tokens | Sum API seconds | Mean seconds |
|---|---:|---:|---:|---:|---:|
| customer | 51 | 77,486 | 1,991 | 228.48 | 4.48 |
| service | 74 | 550,907 | 14,299 | 386.32 | 5.22 |
| evaluator | 2 | 18,003 | 214 | 13.44 | 6.72 |
| customer_evolver | 1 | 23,552 | 6,710 | 82.23 | 82.23 |
| service_evolver | 1 | 23,214 | 7,068 | 80.37 | 80.37 |
| reviewer | 0 | 0 | 0 | 0.00 | 0.00 |
| customer_judge | 0 | 0 | 0 | 0.00 | 0.00 |
| service_judge | 0 | 0 | 0 | 0.00 | 0.00 |

合计 **129 calls；693,162 prompt tokens；30,282 completion tokens；总 tokens 723,444；wall-clock 795.33秒（13分15秒）**。6场完整episode，0失败attempt，0resume。并发为1；这个时长不能外推成E20×2或8路并行的性能结果。

## 实际Pro参数的边界

Pro实际HTTP参数记录为 `model=deepseek-v4-pro, thinking.type=enabled`，没有temperature或显式max_tokens。`reasoning_effort=high`保留在调用方runtime args，但命名的`reasoning_effort`字段没有出现在记录的最终HTTP body中。**可以确认Pro开启思考，不能把high档位当成已经验证的控制变量。**配置原有的`reasoning_parameter_verified_by_provider=false`没有改成true；后续正式报告应使用观测到的条件。

## 文件与后续

- `canary-summary.json`：21项检查与统计、实际参数、证据边界。
- `actual-provider-http.jsonl`：129个真实request/response的脱敏wire记录。
- `evolver-calls/`：两个真实Evolver的准确input、parsed output、provider metadata。
- `episodes/`：6份native simulation、records、telemetry及provider logs。
- `checkpoint.json`：原checkpoint的只读快照；manifest中的原始checkpoint/output路径保持不变。
- `alternating-result.json`、`generation-0000.json`、`service-memory/`：完整generation/NO_OP结果。

按本轮小型canary定义可以freeze这版源码。正式E20×2仍需单独启动；大上下文provider稳定性、ADD/UPDATE实际效果以及泛化均没有由此canary验证。保留原始运行目录用于后续显式resume；此处为只读分析归档，不改写frozen manifest。
