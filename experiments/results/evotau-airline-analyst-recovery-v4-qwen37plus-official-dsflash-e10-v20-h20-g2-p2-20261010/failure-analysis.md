# 本轮进化失败的证据审查（2026-10-10）

范围：冻结提交 be651f3 的完整 G2 工件、生成 Prompt、校验器、逐轮 SkillActivator、原生 policy、Screen 与 native reward。全部为离线审查，没有新 API 请求，没有修改运行或算法。可重核的数据在 failure-analysis-evidence.json；原始文件仍在 full-artifacts.tar.gz。

## 核心判断

本轮不是“有一批有效 Skill 被统计 Gate 否决”。没有候选进入 Full-E 或 V Gate。Gen0 没产生合规 Analyst 输出；Gen1 两个候选只有一个进入真实 rollout，而这个候选的关键指导在执行写操作前已从系统提示中撤下。不能据此证明 Skill evolution 无效，也不能归因于 API、预算或 V 的统计门槛。

## 1. Gen0 因契约失败失去了整代 Service 搜索

- 初次 Analyst 调用 0aaf16e010bb4891a121ae2a9d2e6327：两处 evidence hash 抄错。Task12 message7 的原始 hash 含 `...91ac2ad5...`，输出变成 `...91ac2d5...`；Task7 message22 的 `...83fcb906...` 变成 `...83fcbe906...`。
- 第一次恢复 588833ed810c4ce9984d37962b692e01：三个 hypothesis 的 alternative_explanations 均为字符串，而契约要求字符串数组。
- 第二次恢复 5174f908928746f4b45ad7f1bd751719：Task7 message23 hash 的 `...71e39555...` 变成 `...71e31955...`。

严格校验不是误报。根因之一是把长 hash 的逐字复制交给生成模型：分析文本可以有内容，但索引契约不合规导致整批 hypotheses 被拒绝。恢复后继续了实验，却没有形成新的 Skill 修复机会。恢复 Prompt 只加通用“遵守原契约”提醒，没有提供具体哪处 hash 或字段违反了契约。

可考虑短证据 ID + 由程序解析到不可变原始索引/hash；仍严格验证存在性、来源和任务标签，不能静默改模型返回的错误 hash。这是建议，不是本次已实现改动。

## 2. 唯一实际测试的 Skill：做了查价，却在比较预算和写入前失活

目标 Task12：用户要求两位乘客升舱，最多支付 650 美元，并且无论能否升舱都添加两件行李。Baseline 未先查价，升舱扣款 1200 美元，native DB mismatch，0 分。

候选 Skill要求：查价 → 计算总差额 → 与用户上限比较 → 超限不执行。但其 positive_conditions 同时要求“尚无工具结果给出目标舱位价格”。这个条件在流程第一步完成后即为假。

真实轨迹与激活对应：

| Service turn | 当前可见信息 | Skill 指导在系统提示里 | 后续行为 |
|---|---|---|---|
| 3 | 用户表达升舱和 650 上限 | 是 | 查第一段航班 |
| 4 | 已查第一段，第二段未查 | 是 | 查第二段航班 |
| 5 | 两段 business 价格 350 与 499 已知 | 否；系统提示 hash 等于原生 | 讨论 cabin/flight pricing 规则，未完成预算约束检查 |
| 6 | 用户 profile 已知 | 否 | update_reservation_flights(business)，实际收费 1200 |

Activator 自由文本还把“不曾查到价格”这条正向条件称为负向条件，分类表述不准确；但停止激活本身符合实际触发器的窄范围。源码 skill_activation.py 每轮重新挑选，将 state.system_messages 重写为 native prompt + 当轮 active guidance；没有跨步骤保留未完成义务。之前的查价行为在对话中可见，但整个 Skill 的余下规则已不在 system prompt。

因此可以直接观察到“只执行了修复的第一步”。撤下指导是否是唯一充分原因，未通过对照干预证明；这是证据充分的机制缺口，不是已经完成的因果实验。

随后用户要求撤销超预算升舱。候选把 cabin 改回 economy，但支付历史新增 1200 扣款和 -1170 退款，两段 economy 价格也从 122/127 变为 125/139。native DB match 仍为 false。不是“最后显示 economy 就应该成功”：该任务 reward_basis 为 DB + COMMUNICATE，action checks 的有无并非本轮单独的计分依据。

Screen六个条件：五个成功保持成功，Task12失败保持失败；0修复、0回归，故未通过。这次不是被随机回归卡住。

建议先修候选的触发生命周期：未完成的价格/预算/授权检查须持续到承诺动作前完成，或明确取消该操作，不能只覆盖查价开始阶段。无需先放宽 Screen/V。

## 3. 另一个候选分析了疑似问题，却把自己的政策判断写成新禁令

Task7 hypothesis 把“basic economy先升business再取消”定性为规避政策。候选禁止为了满足取消条件而升舱，语义校验给出 policy_subordinate=false。

固定原生 policy 的 Change cabin 段允许未飞的 basic economy 换舱；Cancel flight 段包含 business flight 可取消；没有单列“以取消为目的不得升舱”。这不证明整个 Task7轨迹正确（仍可能有授权、其他状态或任务目标问题），但说明不能从失败 reward 单独推出模型新增的目的性禁令。Analyst自身 alternative_explanations 已承认字面政策可能允许该序列，Mutator仍将假设写成禁止规则。

应基于可观察证据提出修复，并以明确 policy 条款检查；“看着像规避”不是修改原生业务规则的权限。拒绝这条候选有实质依据。

## 4. Customer两代都没有形成有效挑战

两代候选均被语义校验拒绝，而未真实评测：它们要求模糊/预先授权、隐藏预算至操作后，再增加附带请求或争议。Gen0甚至使用通用 upgrade-then-cancel 指令，无法对所有任务保留原始目标。最终 fresh Customer同样拒绝。审查器没有把候选当0分任务，也没有偷偷部署。

所以本轮两代 Customer和已部署Service都未变化；初始E只有2/10失败，第二代沿用相同E成绩。两个失败案例可支持少量假设，但无法当成广泛可复用机制的证据。候选多样性与交替压力均有限，不能把G2理解为已经完成两次有效改进。

## 外部源码对照

- PRISM官方 proposer读取失败与成功对照、历史和failure matrix；prompt-only更新整份 system_prompt.txt，middleware变体另有前置条件拦截。它与本项目每轮可撤下的 Skill 指导不是相同执行机制，不能照搬生成 Prompt后假设有相同效果。https://github.com/airbnb/agent-harness-optimizer/blob/main/agent_harness_optimizer/optimizers/prism/proposer.py
- GEPA官方reflective proposer把轨迹和反馈用于候选生成，候选需重新评估；不能把看起来合理的反思等同于效果。https://github.com/gepa-ai/gepa/blob/main/src/gepa/proposer/reflective_mutation/reflective_mutation.py

外部方法仅用于对照；本报告的故障结论来自本轮原始工件和冻结代码，不来自论文排行榜。

## 优先级与证据限制

先处理证据引用机械负担和多步骤Skill触发生命周期，再改善policy-grounded假设/Customer目标保持。不是换API、扩大输出额度、放宽V门槛或直接加更多代数。本轮732次请求HTTP均成功，实际测试候选均正常终止。H结果不用于修复反馈；无需加载额外H内容。当前不能证明始终保持Skill就必然修复Task12、不能宣称DeepSeek能力不足、不能证明总体进化方法有效或无效。若后续用户批准，应单变量验证生命周期修正，保留native evaluator与最终状态判定。
