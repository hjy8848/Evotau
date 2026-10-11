# 人工 Customer / Service Skill 真实对照结果

**两条人工 Skill 没有改善这批 E3 的 Native Accuracy，Customer 也没有增加 Native 失败。** 这次提供的是一个负结果与具体执行证据，不是高质量 Skill 已经有效的证明。

冻结代码 `4753adfcfa7ed57b5da4f59e2a59a78501630408`；配置 `configs/airline-human-scope-ledger-v1-e3-factorial-p2.yaml`；Manifest SHA `2e355b8095427a7590b6a709b4aac18389e00ffd7ba51d9a5eee8cb4cb72c254`。

四组独立执行 Task 7、12、15，seed=1，max_steps=200，panel 内并发2。模型均为网关 `dashscope/qwen3.7-plus`，temperature=0，enable_thinking=false。没有 Evolver 请求，没有 V/H、评分导入、Skill 晋升。原生 evaluator 实际完成评分；本组 DB/COMMUNICATE 检查无需 evaluator 模型请求。结构校验通过，但没有调用付费语义 Validator，人工后审不能冒充完整正式验收。

| 条件 | Task 7 | Task 12 | Task 15 | Native Accuracy |
|---|---:|---:|---:|---:|
| 原版 C0S0 | 0 | 0 | 1 | 33.3% |
| 人工 Customer C1S0 | 0 | 0 | 1 | 33.3% |
| 人工 Service C0S1 | 0 | 0 | 1 | 33.3% |
| 两边人工 C1S1 | 0 | 0 | 1 | 33.3% |

各组分解也相同：Task7 DB=1/COMMUNICATE=0，Task12 DB=0/COMMUNICATE=1，Task15 两项均为1。12场均正常 user_stop，没有 max_steps 截断或 provider 失败。小样本单 seed、独立自然语言对话不能用于严格因果归因，也不能用于泛化结论。作者看过 E 评价定义，不是盲测 Evolver 能力。

## 已核实的执行问题

1. **触发得太晚。** C0S1 Task12 中，Activator turns 0–3 均选择空 Skill，turn2 的理由是“information gathering”，未把用户的升级要求视为 pending write。随后 Service 在原生 messages[6] 调用 `update_reservation_flights`，结果 messages[7] 显示新增收费1200。Skill 直到 turns7/8 用户要求纠正后才激活。此前的 $650 限制已出现在用户对话中。不能把这段违规解释为“用了修复 Skill 仍然无效”，它在关键第一笔扣款前根本没注入。
   - 原始：[native-simulation.json](episodes/1daab0accae74dfe943c034da51f8862/native-simulation.json)
   - 激活：[skill-activation-trace.json](episodes/1daab0accae74dfe943c034da51f8862/skill-activation-trace.json)

2. **Customer overlay 没有保证任务约束落实。** C1S1 Task12 用户 messages[1] 提出升级与加行李，messages[9] 批准升级，但整场没有披露 $650。messages[11] 工具结果实际收1200，messages[17] 用户接受并结束。原任务明确愿付上限650，超出则尝试同行者升级；本轨迹不保留该约束。人工审查将它标为 INVALID，不作为合法挑战或修复证据；原始 Native0 保留不改。只是程序性“yes”更清楚，不能称挑战成功。
   - 原始：[native-simulation.json](episodes/bda65e50ea9f48f1b08600c041a4b8fa/native-simulation.json)，messages[1,9,11,17]。
   - 此例 turn4 确实注入 Skill（408 proxy tokens，prompt hash 与 native hash 不同），但执行前仍无明确价格。说明“有 Skill”不自动等于满足业务前置条件。由于用户约束也丢失，不能隔离哪一项导致结果。

3. **人工挑战的新压力不足。** C1S0 Task15 的机场替代请求仍是原场景已有机制，Service 正确拒绝目的地变更并完成原操作。Task12 的预算纠正发生在违规扣款之后；Task7 的信息问题出现在取消操作之后，未形成设想中的操作中断。文字写了触发和忠实性，不代表 simulator 一定执行。
   - C1S0 Task15 messages[17,23]：[轨迹](episodes/4dd288affb164a36b90891f6df10fdf9/native-simulation.json)。
   - C1S0 Task12 messages[15]：[轨迹](episodes/1ef7a79af1654178b1425c4d054e1c4e/native-simulation.json)。
   - C1S0 Task7 messages[17]：[轨迹](episodes/e6d7cf8a41674b8eb03e544907a8b118/native-simulation.json)。

Task7 在所有条件下的总分仍由 COMMUNICATE0 决定。既有审计已指出该静态目标与问题时序的错位；本次没有用答案补丁制造提升。这批实验不适合用 Task7 总分检验授权 Skill 的有效性。

## 激活与用量

C0S1：Task7激活2轮、Task12激活2轮、Task15激活2轮。
C1S1：Task7激活0轮、Task12激活1轮、Task15激活1轮。
这进一步说明同一 Skill、同 task×seed 也可能有不同暴露条件。

| Call name | Requests | Input tokens | Output tokens |
|---|---:|---:|---:|
| Service agent_response | 157 | 1,344,485 | 42,891 |
| Customer user_simulator_response | 66 | 113,545 | 3,284 |
| Skill Activator | 75 | 307,197 | 8,274 |
| 合计 | 298 | 1,765,227 | 54,449 |

实际耗时862.20秒（14分22秒），298成功/0失败。费用未知，不猜人民币价格。已通过98项相关离线测试，Ruff通过。

## 下一步

先离线修正**“用户已经要求受约束的写操作，但 Activator 仍认为没有 pending write”**这一激活定义，并在保存的 Task12 turns0–2 上验证预期触发。不要在本冻结运行内改 Skill、重算成绩或直接继续大规模收费。新的触发候选必须有独立版本，再测试其真实执行情况；本次没有自动追加实验。

`result.json` 是未经修改的运行汇总；`episode-summary.json` 提供四组索引。全部12条原生完整轨迹和激活日志在 `episodes/`，SHA在 `archive-provenance.json`；原始提供商元数据在 `actual-provider-http.jsonl`。本归档不包括所有付费响应缓存，完整运行目录保留该缓存。
