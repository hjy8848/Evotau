# Retail 五轮自进化：完整真实运行结果

**五轮执行完成，但本轮没有验证到有效进化。Customer 候选均未进入 rollout，Service Analyst 的结构输出错误阻断了所有 Skill 生成；零晋升不能解释为修复 Skill 无效。**

| Gen | Customer before | 候选0/1 | Customer after | Service E before/after | 晋升 |
|---|---:|---|---:|---|---|
| 0 | 80% | 未评测 / 未评测 | 80% | 80% / 80% | 否 |
| 1 | 80% | 未评测 / 未评测 | 80% | 80% / 80% | 否 |
| 2 | 80% | 未评测 / 未评测 | 80% | 80% / 80% | 否 |
| 3 | 80% | 未评测 / 未评测 | 80% | 80% / 80% | 否 |
| 4 | 80% | 未评测 / 未评测 | 80% | 80% / 80% | 否 |

## 最终评测

- Native H：S0 = ST = 80%（32/40）；20 tasks × seeds 1/2。Service 身份相同，ST 原样复用 S0，不重新生成对话。
- Pass¹ = 80%，Pass² = 70%；只有两个独立运行条件，不报告 Pass⁴。
- Fresh-adaptive H：未评测。Fresh Customer 被语义校验拒绝，没有用原生 Customer 伪装 fresh。
- V：配置启用，但没有候选到达 V，因此本轮 V 未评测。
- E：16/20 = 80%。相同条件基线在后续四轮缓存复用，不能把五个 80% 当成五次独立重复测量。

## 阻断原因与证据

- 十个 Customer proposals 全被拒绝：原始原因和 semantic review 见 `run/generation-0000.json` 至 `generation-0004.json`。其中包括任务忠实性冲突和非法/不在供应证据中的引用；拒绝不等于候选经过实际对话后没有发现能力。
- 每轮 Analyst 首次请求加两次冻结格式恢复，共15次，全部未通过 schema。原始 response 在 `run/evolver-calls/`，各轮状态在 `run/evolution-v2/g000N-service_failure_analysis.json`。
- 可确定的契约问题：`failure_analysis.py::validate_analysis` 要求 `alternative_explanations` 为字符串数组，真实输出使用字符串；`FAILURE_ANALYST_PROMPT` 仅列字段名，未明确数组类型。没有静默转换或放宽校验。
- 因此 Mutator、Screen、E/V Gate 没有实际执行；不能说模型找不到好 Skill 或本场证明修复无效。Customer validator 与原场景矛盾行为的相容性仍需要单独审核，拒绝原因是模型审查判断，不能默认均正确。

## API 和时间

- Provider：1,290 次请求，1,290 成功，0 HTTP/API失败；schema拒绝另行记录，不计为网络失败。
- 输入 8,428,360 tokens；输出 229,134 tokens。
- 总 wall-clock 2199.11s（36.65分钟）；evolution 1272.71s；H 926.44s；V 0s。
- 实际完整 native episodes 60；逻辑episode jobs 140，其中80缓存命中。最高episode及wire请求并发均4。
- 没有新增请求/费用上限；人民币价格未知，费用 N/A。

| 角色（按真实 call name） | Calls | Prompt tokens | Completion tokens | 请求耗时总和(s) | 平均延迟(s) |
|---|---:|---:|---:|---:|---:|
| Customer runtime | 412 | 576891 | 20851 | 840.50 | 2.04 |
| Service runtime | 823 | 6541010 | 162403 | 3846.25 | 4.67 |
| Evaluator | 24 | 171530 | 3326 | 103.58 | 4.32 |
| Customer Evolver | 6 | 256796 | 6497 | 125.55 | 20.93 |
| Service Analyst | 15 | 840494 | 33981 | 632.22 | 42.15 |
| Semantic Validator | 10 | 41639 | 2076 | 47.90 | 4.79 |
| Service Mutator / Activator / Reviewer / Customer Judge / Service Judge | 0 | 0 | 0 | 0 | — |

原始 `api_usage_by_role` 将 Customer Skill Evolver 分在 other；本表按真实 call name 归类，未更改原始账本。并发下请求耗时总和不是 wall-clock。36.65分钟也不能作为相对旧一小时实验的严格加速比：本轮候选没有进入评测，工作量不同。

## 冻结身份与限制

- 代码：`9bb79be28d41d431b6be7ce2d2e95b5f78de6495`。
- Manifest：`db065b029cf944f622f04b0b86eccb3a78c8e208997d1c3ab51b482bb3dd0273`。
- 原生 τ² Retail pinned b7ea9074、max_steps200、全角色 gateway Qwen3.7-plus、temperature0、thinking关。
- E20/V20/H20、五轮、四路并发、既有 Analyst/Customer Prompt和Gate；未修改原任务、policy、tools、评分、历史冻结配置。
- A/A未校准，H有限面板，不能宣称总体泛化；同task×seed不等于同实际对话。
- 完整原始JSON、native-simulation、provider元数据、immutable stage和失败证据保存在 `run/`，文件SHA索引在 `archive-audit.json`。
- 本轮主要证明流程和缓存可完成；有效的Customer压力与Service修复仍未验证。最小后续工程修正应先明确Analyst字段类型并补契约测试，使用新冻结版本，不改本轮结果。
