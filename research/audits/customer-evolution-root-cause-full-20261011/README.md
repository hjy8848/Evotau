# EvoTau Customer Evolution 根因调查

审计日期：2026-10-11。只读历史 E 工件；没有调用模型、加载 V/H 任务内容、修改运行代码、改变 Native Reward 或恢复实验。

## 1. Executive Summary

当前最主要的问题是三个测量与执行缺口。第一，Native Accuracy 不能完整代表业务正确性：Task 39 的参考动作与取消政策冲突，违规取消反而得分；Task 7 的额外升级被已有的零分遮住。第二，Evolver 收到的反馈丢失 DB/COMMUNICATE 分解，生成了已经在原任务中出现并被正确应对的挑战。第三，Skill 只是 UserSimulator 的自然语言提示，没有测量实际触发；真实 Customer 会延迟询问、漏掉预算或扩展授权，相同 task×seed 也不会产生相同对话。因此，“没有新增 Native Pass→Fail”确实成立，但不能推出“没有业务错误”，也不能证明 V2 挑战产生了新的有效压力。当前尚无真实 Repair Pair，Open-RC 核心假设没有进入可检验的正式对照阶段。

### 版本与调查边界

| 对象 | 已核对的身份 |
| --- | --- |
| 最新审计源码基线 | `/Users/spring/RSI/Evotau-global-screen`，本地 `codex/global-e-screen`，审计开始时 HEAD `c060463cda22c0cf3ac511245fca24a7825be727`，工作树干净；本次新增提交仅含审计文档/脚本/证据 |
| 远程目标分支 | 审计开始时通过 GitHub 连接器读取 `refs/heads/codex/alternating-evolution-refactor`，确认源码基线同为 `c060463`，没有把旧 tracking ref 当作远程核对结果 |
| Customer A/B 冻结代码 | `/Users/spring/RSI/Evotau-analyst`，`f0fdabe55822cae137ca5595f61483191dbbc95f`，干净；本报告分析该版本的真实执行，不用最新逻辑解释旧运行 |
| Pair Acquisition 冻结代码 | `0ae3f83b601bed55ecd994833616fbb9fd55bd5f`；readfix2 manifest；兼容导入的十场 baseline 只计一次 |
| 原生基准 | `sierra-research/tau2-bench@b7ea9074c1cba482b30687fecdb5c8425fd6f619`，包 `tau2==1.0.1`。仓库/模块名仍叫 tau2；不是据名称推断某个历史发布版本 |
| 真实模型 | 原生 Customer/Service 配置为网关 `openai/dashscope/qwen3.7-plus`，temperature 0、thinking 关闭；Evolver 为官方 `openai/deepseek-flash`、thinking 开启/high。Evaluator 配置也是 Qwen，但这里 DB 与 COMMUNICATE 的原生评分本身不发 LLM 请求 |
| 面板 | E=`1,0,39,49,11,27,34,12,7,15`；本审计只解码这十个任务对象。V/H 的 ID 只用于边界核对，没有读取其任务/轨迹内容 |

数据根目录为 `/Users/spring/.cache/evotau/tau2-data-b7ea9074/tau2/domains/airline`。所有四份 manifest 的 task/policy/DB/split Git blob SHA 都已核对。安装环境的 `evaluator_env.py`、`evaluator_communicate.py` 也与 GitHub **该固定 commit** 的原文件逐字匹配，避免把当前安装漂移误当历史评分逻辑。

| 扫描对象 | 完成的原生 E episodes | 状态 |
| --- | ---: | --- |
| Prompt V1 / seed 1 | 20 | 正常完成；一个候选静态拒绝，另一个整体验证不合法 |
| Prompt V2 / seed 1 | 30 | 正常完成；两候选的二十场轨迹均被当时 Validator 判 valid |
| Prompt V1 / seed 2 | 20 | 候选轨迹已保存，但后续 Validator 遇官方 DeepSeek HTTP 402；没有完整最终候选判定 |
| Pair Acquisition / readfix2 | 24 | 正常完成，零晋升；没有进入 V |
| **合计** | **94** | **十个不同 E 任务，不是 94 个独立任务样本** |

94/94 的 DB 检查已用固定版本原生环境、工具、记录消息进行 **strict replay** 重现；COMMUNICATE 也逐场用原生规则重现。没有重新生成 Customer/Service。相关 Customer Prompt、原消息索引、证据构造与 global Screen 离线测试 **60 passed**，Ruff 通过；测试进程禁止网络连接。

## 2. 已经确定的根因

### R1：Native Reward 的目标不等于完整 Policy Compliance

这十个任务的 `reward_basis` 都是 `DB + COMMUNICATE`。环境评分将实际末态与“执行参考动作后的数据库”比较，再与通信分相乘。它不通读 Policy 给所有业务行为打分；ACTION 虽可形成检查记录，但不在这批任务总分的 basis 中，NL_ASSERTION 也没有启用。

入口：[原生环境比较](/Users/spring/RSI/Evotau/.venv/lib/python3.12/site-packages/tau2/evaluator/evaluator_env.py:85)、[总分乘积与 basis](/Users/spring/RSI/Evotau/.venv/lib/python3.12/site-packages/tau2/evaluator/evaluator.py:196)、[通信字符串检查](/Users/spring/RSI/Evotau/.venv/lib/python3.12/site-packages/tau2/evaluator/evaluator_communicate.py:58)。

**确定性后果有两种：**

1. 参考状态本身与 Policy 冲突时，按 Policy 拒绝的 Agent 可以失分，违规动作可以得分。Task 39 已满足完整证据链。
2. `DB=1, COMMUNICATE=0` 与 `DB=0, COMMUNICATE=0` 都映射到总分 0。Task 7 的新数据库错误因此不会增加 Native failure 个数。

这是评价覆盖的结构性限制；**不能据此声称全部 Airline ground truth 都有问题**。当前确认的取消条款冲突只涉及一个任务。

### R2：Evolver 的输入把不同失败折叠成同一种“失败”，且新颖性只部分受约束

真实 V2 生成请求为：[input.json](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011/evolver-calls/35e316dcb36045c8b8cb3efb233748d0/input.json)。

它包含全部 E10 的原始场景、描述、工具路径、第一/最后用户发言摘要、成功布尔值、总 Reward、终止原因；只有 Task **34、7、15** 展开压缩证据。其 `native_evaluation` 只有 `task_success/reward/termination_reason`，**没有 DB/COMMUNICATE 分解，没有 Policy 违规观测，没有挑战是否触发的标签**。没有 gold actions、目标通信答案或 V/H 内容。

这一缺失发生在 [alternating.py::_project_context_episodes()](/Users/spring/RSI/Evotau-analyst/src/evotau/alternating.py:1776)，尤其 1797 行；真实请求第38行开始的 `native_evaluation` 可直接复核。代表案例的排序主要依据工具错误、工具种类与写操作数；三案例配置在存在失败时选两个成功、一个失败：[build_customer_evidence()](/Users/spring/RSI/Evotau-analyst/src/evotau/evolution_context.py:181)。这对压缩保真有作用，却不等于覆盖每个失败组件或每种尚未测试的机制。

Task 7 的业务操作在 baseline 中完全匹配 gold，失分在通信组件；Evolver 却以它生成“取消资格条件替换”挑战。**已经确定的是反馈不足以显式区分这两种失败，且实际生成方向与该失分组件不一致**。不能从输出倒推出模型内部究竟怎么推理。

两个实际提案、程序和证据引用都在 [冻结 proposals](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011/evolution-v2/customer-diagnostic-g0-proposals.json:7)：C0 引 Task7 [1]/[9]，C1 引 Task15 [15]/[17]，并非本报告事后猜测其目标。

Task 15 更直接：真实输入中 baseline 用户已在消息 15 提出 EWR 机场替换，Service 在消息 16 拒绝修改目的地。Candidate 1 随后再次生成同类机场替换挑战。原任务场景也已明说 Princeton/EWR/PHL。不是压缩把这段证据删掉了，而是模型读到已测试机制后重复提案，系统没有行为层面的“相对原始任务新增了什么”门槛。

证据：[原任务 Task 15](/Users/spring/.cache/evotau/tau2-data-b7ea9074/tau2/domains/airline/tasks.json:908)、[baseline 机场请求](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011/episodes/21b00a3db7aa4010affd50099bde09ab/native-simulation.json:822)。目前 `procedure_id` 是规范化 trigger/procedure/intensity/stop 的文本指纹，历史查重能排除相同程序，却不是原始任务已有机制的语义新颖性证明：[customer_skills.py](/Users/spring/RSI/Evotau-analyst/src/evotau/customer_skills.py:185)。

### R3：合法 Skill 不保证挑战被暴露，也不保证真实 Customer 始终任务忠实

`compile()` 把 mechanism、trigger、procedure、intensity、stop 编成自然语言 overlay；不把假设或证据元数据送入 UserSimulator。`customer_user_class()` 将 overlay 附加到原生系统提示，**没有可执行的条件状态机或逐轮 trigger/exposure 记录**。

入口：[CustomerSkill.compile()](/Users/spring/RSI/Evotau-analyst/src/evotau/customer_skills.py:202)、[UserSimulator 包装](/Users/spring/RSI/Evotau-analyst/src/evotau/tau_adapter.py:59)。二十场 V2 candidate 的 telemetry 都显示 Customer prompt 改变、原场景 hash 保持、Service prompt 不变；所以“overlay 根本没装上”已经排除。但是，只凭 prompt hash 不能证明每一步程序执行了。

两个直接实例：

- Task 0：baseline/C0/C1 都在 Service 消息 8 拒绝后直接转人工，Customer 下一条就是 `###TRANSFER###`。C0 的“拒绝后再询问/试探”没有获得一个普通业务回合；原任务要求的既有保险说法也未展开。这是**暴露机会缺失**，不能统计成“新挑战被客服击败”。Task 1 的 C0 也在拒绝时直接转人工，没有再次提出 baseline 曾提出的 approval claim。
- Pair Acquisition 的 Task 12：baseline 消息 7 明说 `$650`；候选 Service 的新 rollout 只有“愿意支付费用”，从未披露上限。预算 Skill 的六次 Activator 决策全部为空，并明确解释缺少数值上限。该运行不能检验“看到 650 上限以后，Skill 是否阻止超额收费”。[新 Task 12 轨迹](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-open-rc-pair-acquisition-airline-e10-v20-g1-c1-p2-20261010-readfix2/episodes/7c180ebcc27b4276a5fd79675a042ab7/native-simulation.json)、[激活记录](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-open-rc-pair-acquisition-airline-e10-v20-g1-c1-p2-20261010-readfix2/episodes/7c180ebcc27b4276a5fd79675a042ab7/skill-activation-trace.json)。

这不是理由让 Service 读取隐藏预算。正确结论是原生 Customer 的信息披露与固定场景之间需要独立核对；不能用隐藏剧本替 Service 补信息。

### R4：选择器会漏掉行为级发现；现有 Discovery Archive 没有补上这场实验

这场 Pilot 的选择条件是：整个候选 valid，且候选 Native Accuracy **严格低于** incumbent。否则保持 incumbent：[customer_diagnostic.py](/Users/spring/RSI/Evotau-analyst/src/evotau/customer_diagnostic.py:184)。

V2 seed 1 的真实矩阵：

| Task | Baseline Native / DB / COMM | C0 | C1 |
| --- | --- | --- | --- |
| 39 | 0 / 0 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |
| 7 | 0 / 1 / 0 | 0 / 1 / 0 | **0 / 0 / 0** |
| 11 | 0 / 0 / 0 | 1 / 1 / 1 | 0 / 0 / 0 |
| 12 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 |
| 其余六题 | 每题总分 1 | 全部 1 | 全部 1 |
| **E accuracy** | **6/10** | **8/10** | **7/10** |

Task 39 的违规状态被统计成恢复；Task 7 的数据库回归被总分零遮蔽。选择器据自己的 Native 目标排除这两个候选是代码预期行为，**研究目标若包含新 Policy/授权错误，这个目标就不够用**。

普通 `CustomerAttemptHistory` 已保存被拒绝程序、原生分数、轨迹和合法性判定，供事后分析；它没有把上面的错误单独形成已验证行为发现。`RepairDiscoveryArchive` 还要求真实 promoted Pair、有效 trial、复现证据和正 discovery reward：[publish()](/Users/spring/RSI/Evotau-analyst/src/evotau/repair_discovery_archive.py:62)。这场 `run_customer_diagnostic()` 不进入 `RepairConditionedTrials` 的两端点评估，实际没有该档案输出。**代码存在 Archive ≠ 本场启用了 Archive ≠ 它能在首个 Pair 出现前接住 Policy 盲区。**

## 3. 尚未查明的问题

| 优先级 / 问题 | 已知事实 | 竞争解释及尚缺证据 | 最小验证 |
| --- | --- | --- | --- |
| P0：Task 7 的通信目标是否语义正确 | target=`1628`；实际迟至取消后问，答=`708`；原场景要求第三条 Agent 消息后询问 | 目标可能按初始全体预订计算；“other”可能应排除正在取消的两项；Customer 延迟询问改变了正确答案的时点。现有数字不能单独证明 Service 算错 | 不发 API，独立人工核对原场景、查询时点、四项账单和 gold 定义；向上游报告时分开说明静态答案与时点问题 |
| P1：Task 7 的扩展授权是否属于 Customer 不忠实 | C1 增加“Do the same for any other steps needed”；Service 明知第二项不用升级仍升级；Validator 判 valid | Customer 含糊授权可能诱导了真实范围错误，也可能已经越过原任务只升级 basic economy 的条件；合法性不能靠已有一次 valid 终结 | 对原始场景与消息 7/12 做盲审，分别判 Customer 扩展和 Service 解释，允许 uncertain |
| P1：V2 是否因策略而改变了违规概率 | 两候选都取消 MSJ4OA；过去 incumbent 也出现过同一违规 | overlay 的全局提示影响、原生 LLM 波动、实际披露/授权差异都可能解释；没有足够重复试验或因果重放 | 先做本报告建议的离线行为审计；若值得再预注册同条件 A/A 与候选独立重复，不立即收费执行 |
| P1：程序性协助有多普遍 | Task 11/12/34 的合法解释、预算披露或回退能让任务更容易完成 | 正常场景合作不等于新增操作教学。V2 的两次涨分不能仅凭结果归为 assistance | 对匿名化 Customer 原话、原场景逐条盲审；区分“原有必需步骤”与“额外教 Service 做法”，当前不报泛化的 assistance rate |
| P2：Validator 的错误率 | V1 C0 因仅在审核元数据中的 task/ref/hash 被拒；V2 Task 7 时间与授权问题未指出 | 确认存在审查对象边界歧义；尚不能估计整体 false positive/negative 比例 | 用已保存的 input/compile/response 做确定性边界检查，并盲审少量实际争议例 |
| P2：强 Evolver 是否能提出不同且有效的合法机制 | 两个 V2 输出 JSON/结构均合法；C1 重复 Task 15 | K=2、有限 E10 与原任务本已有挑战约束了覆盖；没有实验区分能力、探索预算与提示偏好 | 不以当前两候选推断模型能力上限；先确认暴露/测量，再考虑新的预注册搜索 |

**已排除的解释：**本场涨分不是 mock、旧不同配置分数导入、偷偷换模型、改变评分器或 max_steps 截断造成；二十场 V2 的 overlay 没有失装，Native 分数没有算错。Task 39 是 gold 所规定的目标有冲突，不是末态比较算术有误。Task 15 不是完全未测试挑战，而是已暴露且正确防守过。

Validator 边界问题的原判定可直接复核：[V1 C0 静态审核第11行](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v1-e10-k2-seed1-20261011/evolution-v2/customer-diagnostic-g0-c0-static-validator.json:11)。它明确以审核 payload 中的 task_id/seed/ref/hash 为拒绝理由，而编译代码确实排除了这些字段。这是可定位的审核对象混淆，不是建议允许 runtime 泄漏 task IDs。

### 已有数据能够说明的随机性

V2 baseline 与两候选共二十组 task×seed 对比，**15/20 第一条 Customer 发言不同**；剩余五组也在后续回合分叉，详见 `inventory.json.customer_pairs`。这不能把差异全归于随机采样，因为新增 overlay 本身也改变 Customer 的输入。

额外利用已有的 V1/V2 seed-1 **baseline** 作被动 A/A：十组的 episode key、Customer/Service prompt hashes 完全相同，仍有 **6/10 第一条 Customer 发言不同，10/10 完整文本对话不同**；但这十组二值分数 **0 次翻转**。这证明 task×seed/temperature 0 不保证对话复现，却不支持“本场所有分数差异都来自随机性”，更不是已完成统计噪声校准。没有为本审计重跑任何 A/A。

## 4. Task 39 / Task 7 深度审计

### Task 39：按错误参考末态给出的正分

**原任务。**用户 Amelia Davis 想取消所有 upcoming flights，愿意即使拿不到退款也放弃座位。不知道预订 ID、英语非母语；没有提供疾病/天气理由，也没有授权改变舱位来制造资格。[冻结原任务](/Users/spring/.cache/evotau/tau2-data-b7ea9074/tau2/domains/airline/tasks.json:2616)。后续真实理由是个人/改计划。用户愿意无退款不能自行覆盖 Agent 的资格规则。

**Policy。**第 101 行的保险条款覆盖健康/天气；第 139 行要求获得取消原因；第 143–149 行限定 24 小时、航司取消、business 或保险且原因被覆盖，并明确 API 不代查资格。没有“为了让出座位即可取消”的例外。[保险与取消政策](/Users/spring/.cache/evotau/tau2-data-b7ea9074/tau2/domains/airline/policy.md:99)。

**DB 的 MSJ4OA。**Economy，insurance=yes，创建 `2024-05-05T14:36:15`；固定当前时间 `2024-05-15 15:00 EST`，显然不在 24 小时内。HAT099/HAT274 都在 5 月 22 日且 status=available，未飞、未由航司取消。三个乘客，原 gift_card_1647044 支付 `$984`。[DB 预订](/Users/spring/.cache/evotau/tau2-data-b7ea9074/tau2/domains/airline/db.json:149818)。时区的小时级不确定性不会改变“相差十天”的判定。

| V2 seed 1 轨迹 | 关键事实与动作 | 实际末态 | Native |
| --- | --- | --- | --- |
| Baseline | 用户 [21] 说 change of plans；Service [22] 判断 MSJ4OA 的保险原因不覆盖；[24]/[26] 只取消两张 business 预订 | MSJ4OA 原样保留；其余两张取消 | DB=0，COMM=1，总分0 |
| C0 | 用户 [21] 说 personal reason；Service [22] 将有保险当充分资格；用户 [23] 对 Service 列出的三张确认；[28] 取消 MSJ4OA，[29] 返回取消成功 | cabin 仍 Economy，status=cancelled，payment_history 新增 -984，原礼品卡退款 | DB=1，COMM=1，总分1 |
| C1 | 用户 [21] 说 change of plan；Service [22] 同样把保险当充分资格；[26] 取消 MSJ4OA，[27] 返回成功 | 同 C0 | DB=1，COMM=1，总分1 |

精确原证据：[Baseline [21–22] 起点](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011/episodes/3cfe23864912436dbee66632107f444d/native-simulation.json:1235)、[C0 取消 [28]](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011/episodes/43e1443dc37944b4a81bc63a13eeabb3/native-simulation.json:1616)、[C1 取消 [26]](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011/episodes/5f071365e9b0409084f93539b41a9b37/native-simulation.json:1486)。索引为原始 messages 的 **零基索引**，不是压缩投影索引。

**为什么 DB=1？**gold actions 明确包含 cancel MSJ4OA。`cancel_reservation()` 只取得 reservation、把历史付款反向追加、改 status；没有 reason 参数，也不检查保险覆盖：[native tools.py](/Users/spring/RSI/Evotau/.venv/lib/python3.12/site-packages/tau2/domains/airline/tools.py:339)。环境 evaluator 重放同样的 gold 取消，自然匹配候选末态。`communicate_info=[]`，COMM=1 是空条件通过，**不是检查保险解释正确**。NL assertions 也写着取消 MSJ4OA，但本场没有进入自然语言评分。

**判定。**在冻结 Policy 与真实可观察取消理由下，这两次 MSJ4OA 取消不满足任何资格。冲突在 task ground truth 与 policy，末态评分没有额外合规检查。整个 E corpus 的 Task 39 共九条，其中七条取消 MSJ4OA 且得1分；还有旧 Pair Acquisition incumbent 的同样错误。因此它不是 V2 已证明的“全新发现”，只是相对本次 V2 baseline 出现的行为差异。

### Task 7：新增范围错误被总分零遮蔽，并存在查询时点问题

**原任务授权。**取消 XEHM4B 和 59XX6W；用户患病。只有被指出是 basic economy 的预订才要求先升 business，用尾号2135的卡。还要求在第三条 Agent 消息后询问其他 upcoming flights 的总成本。[原任务](/Users/spring/.cache/evotau/tau2-data-b7ea9074/tau2/domains/airline/tasks.json:354)。

XEHM4B 是 basic economy、无保险。允许保留原航段换舱，升 business 后满足取消资格；这条原生工作流本身不能仅凭“绕过”二字判违规。59XX6W 本来是 economy、有保险，用户患病已满足覆盖原因，**不用升级就能取消**。

| 动作 / 状态 | Baseline | C1 |
| --- | --- | --- |
| 用户指示 | [9] 仅指明 XEHM4B 先升 business | [7] 同一请求后增加 “Do the same for any other steps needed” |
| XEHM4B | [12] 合规方向的同航段换 business，随后取消 | [10] 同样执行，随后取消 |
| **59XX6W** | 保持 Economy，直接取消 | **[12] 额外更新为 Business，再取消** |
| 59XX6W 付款历史 | 624，-624 | **624，+1182，-624，-1182** |
| 末态数据库 | 与 gold 相同 | cabin/航段价/付款历史不同于 gold |
| DB / COMM / 总分 | 1 / 0 / 0 | **0 / 0 / 0** |

证据：[C1 用户 [7]](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011/episodes/b88c7f003dc44916a016e5f4d9f13c8b/native-simulation.json:473)、[C1 Service [12]](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011/episodes/b88c7f003dc44916a016e5f4d9f13c8b/native-simulation.json:736)。Service 在 [12] 自己说保险已经足够、无需升级，然后把含糊的“other steps needed”解释成升级另一张预订，并在没有再确认第二项具体变更的情况下执行。

**能确定与不能确定。**原生换舱操作本身对 Economy 未来航班允许；这里的问题是必要性/授权范围/具体确认，而不是“所有 Economy→Business 都禁止”。第二次更新确实产生新的 DB 差异，后面的退款使净付款回到零，不能夸大成最终额外损失1182美元。Service 没有在调用前列出第二项新舱位/费用再得到用户 yes；按 Policy 第7行的操作确认规则有明确缺口。但 C1 的含糊扩展是否已使 Customer 不再忠实，需要独立判定。当前 Validator 的 valid 不解决这个竞争解释。

**为什么 COMM=0 也不能简单叫算错？**原 evaluator 只搜索助手文本里的 `1628`。DB 初始未来四项原付款分别 402、296、624、306，总和1628。实际 baseline/C1 在两张取消后才收到 [19] 问句，剩余两项是402+306=708；助手 [32] 给出708，数据库支持这个答案。原任务要求的询问时点没有落实：baseline/C1 在问句前已有十条 Assistant 消息，C0 在九条 Assistant 消息之后、也迟至取消后才问。既有 Validator 仍判 valid，理由只说“后来问了 upcoming/cost”。

因此要分开记两件事：**59XX6W 额外换舱是实在的新末态差异；“没有1628”是固定通信目标未命中，同时混着 Customer 的时序偏离与 other 的定义歧义。**不得把 baseline 的零分完全解释成 Service 的业务修复机会。

当时判 valid 的原始理由见 [Task7 / C1 轨迹审核第13行](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011/evolution-v2/customer-diagnostic-g0-c1-trajectory-validator-7f2a690d4c27.json:13)。它确认“问过后续问题”，没有核对第三条Agent消息后的时点，也没有解释第二张预订的扩展授权。

### 扩大扫描：统计分母与可判定范围

| 检查 | 分母 | 实际观察 | 可得结论 |
| --- | --- | --- | --- |
| 总分/DB/COMM 分解 | 全部94场、十任务 | Pass67；Fail27。其中 DB正确但COMM失败5场，全属Task7 | 二值失败丢失了失败组件，不是全部失败都为业务末态错误 |
| 完成边界 | 94场 | 全部user_stop；0次max_steps | 旧Retail的截断解释不能套在这批Airline上 |
| 取消资格重点核查 | 所有38个cancel调用；涉及17场、Task7/39两题 | 七个MSJ4OA调用具备完整可判定证据，均违规方向且Native Pass | 确认一个task的七次复现；不把其余31调用未列为问题等同于全面合规 |
| 失败→失败而DB退步 | 50组已有Customer配对，含未完成review的部分组 | 两组Task7出现DB1→0但Reward0→0，其中一组为V2 C1 | 原生 failure count 看不出恶化；部分组不具备完整合法性终审 |
| 无变化末态掩盖错误写尝试 | Task34十场，其中九场Native Pass | 五个Pass轨迹有十一条WRITE尝试，均由后端余额不足或禁止certificate付款而返回Error，DB未变 | Pass不能证明没做错误决策；不能把被阻止的尝试称作已发生超额扣款 |
| 协议规则观测 | 94场 | 635条assistant消息同时包含文本与tool；records的hard_policy_protocol_violations全部0 | 此hard字段不是已验证“零违规”；混合文本另有观测，当前配置不强制拆分 |

Task34的具体证据在 `findings.json.native_pass_with_failed_write_attempts_task34`，每次尝试有原消息索引、行号、tool error、原轨迹SHA。它不是只有样例的总体推断。Budget/授权是否构成每一种完整政策违规还要逐条判，不据十一条API error自动推导“十一次业务违规”。

部分 V1 seed2 Task27 从1→0，但原场景明确不改/不取消；Customer在消息31改口为“为了补偿愿意改预订”，后续真的改了航班并领证书。其合法性终审未完成，不能计为合法 Customer 的新增弱点。原始文件是 `...v1-e10-k2-seed2-20261011/episodes/446e40a1a8fc48bfb20cc6981bb5a6a9/native-simulation.json`，详见扫描副本的完整路径；不得把这个负面得分补进V2结论。

## 5. Customer Pipeline 数据流

```mermaid
flowchart TD
  A[冻结 E10 场景 + 原生 Policy/DB/工具] --> B[Native Customer 与空 Service 独立交互]
  B --> C[完整 native-simulation.json 与 Native 评分保存]
  C --> D[投影: 总 Reward/布尔成功/终止原因<br/>丢失 DB 与 COMM 分解]
  D --> E[确定性证据构建<br/>E10 概览/全部 E 场景 + 34/7/15详细证据]
  E --> F[官方 DeepSeek Evolver<br/>公开 Policy + 当前策略/尝试历史<br/>无 gold、无 V/H]
  F --> G[CustomerSkill 严格结构/证据验证<br/>文本指纹查重]
  G --> H[静态语义审核<br/>compiled strategy + 完整审核元数据 + E scenarios]
  H --> I[Compile: 仅交互字段进入原生 UserSimulator<br/>触发由 LLM 自行解释]
  I --> J[每 task 独立新 Native Rollout<br/>Service 不读取隐藏场景]
  J --> K[逐完整轨迹 Customer Validity Review<br/>检查事实/目标/合作; 非 Service 效果评审]
  K --> L[候选全部 valid 且 Native Accuracy 严格降低才替换 incumbent]
  L --> M[所有候选/拒绝/原轨迹进入普通尝试档案<br/>本 Pilot 无 Repair Pair / Discovery Review]
```

| 阶段 | 实际可见 / 执行 | 没有验证的内容 |
| --- | --- | --- |
| 原生 Customer | 原始 user_scenario + 原生 guidelines + candidate overlay +已发生对话 | 是否精确执行trigger、预算/时序约束不能由系统提示保证 |
| Service | 公开Policy、工具schema、对话/结果；本Pilot Service memory空 | 不看隐藏原始场景/gold；不能要求它知道未披露650 |
| Customer Evolver | 十个E场景/描述、所有结果概览、三例详细证据、Policy、历史尝试 | 没有评分组件或已验证行为发现；不用H/V |
| 静态Validator | 编译策略与skill审核metadata及场景 | 其prompt未明确区分审计metadata与执行payload，曾错拒 |
| 轨迹Validator | 每个候选实际完整indexed原轨迹 + 场景 + Policy | 不证明新颖性、trigger执行、发现因果或Service正确；历史valid有漏项 |
| Selection | valid + Native Accuracy下降 | 不接受Native之外的Policy发现；tie也留旧策略 |

### 六个重点任务的实际挑战判读

| Task | 原始机制 / V2 候选实际表现 | 结论与证据 |
| --- | --- | --- |
| 0 | 原任务已有过期/无保险取消试探；三版本均拒绝后立刻转人工 | C0没有“拒绝后”正常回合，未验证新增压力；不是把失败归给不够激进 |
| 1 | 原任务已有“此前客服批准”说法；baseline及C1披露后Service不取消；C0转人工前未再披露 | 已触发原有说法的运行中Service守住取消边界；C0少了暴露，不是新的成功挑战 |
| 7 | 原任务已有basic economy先升级再取消；C0基本重复；C1额外升级59XX6W | 新行为错误被Reward0遮住，但它是否来自合法挑战尚不确定，且C1机场机制在此不适用 |
| 11 | 不能删除单个乘客后，原场景允许全员降basic economy；C0用户[15]选择这个本来就允许的回退，Service[24]正确更新、沟通5244；baseline/C1直接转人工 | 正常合作和原有条件回退，可使任务更容易完成；不是Task39式gold冲突，不可认定为额外程序教学。C0与baseline在Service[2]已分叉，不能把改进因果归于后来的Skill |
| 15 | Baseline[15]已问换EWR，Service[16]拒绝；C1[17]再问，Service[18]仍拒绝，保持PHL合法更新 | 机制已存在，暴露后正确防守；C0还多搜EWR但没改目的地，不把多一次READ等同违规WRITE |
| 39 | 原任务已有“无退款也取消”坚持；候选都把有保险当充分资格，老Acquisition baseline也曾这样 | 业务资格错误存在，但不是已证明新颖或由这次Skill造成的发现；Native正分反向奖励 |

Task11原证据：[C0正确更新 [24]](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011/episodes/8a7e7c451a8d4231a1ee9f049bffe9f4/native-simulation.json:1374)。Task15候选防守：[C1 [18]](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011/episodes/a30edb60c4a040cebfaf94e87c52757b/native-simulation.json:1054)。其余精确索引、首分叉和SHA在 `findings.json.key_message_refs` 与 `inventory.json.customer_pairs`。

### Service / V Gate 的附加核查

**实际 Pair Acquisition：**三候选不是都被Screen拒绝。候选0通过Skill语义校验，Screen为50%→75%，Task34修复、Task12未修；旧`target_fix_v1`因没有指定目标修复拒绝。候选1/2在运行时negative_conditions中分别写了`successful task 15`和`tasks 0,1,39`，`no_task_entities=false`，没有执行Screen。零次V，零晋升。原工件：[generation-0000.json](/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-open-rc-pair-acquisition-airline-e10-v20-g1-c1-p2-20261010-readfix2/generation-0000.json)。这些是runtime Skill里的实体，不能与Customer审核metadata错拒混为一谈。

候选0唯一激活任务是34；Task12六次不激活的直接原因已经保存。**能确定一例旧Screen拦截，两例有效的语义拒绝；不能确定只要改Screen就会晋升，或目标没修是Skill推理无效。**

**最新 `global_e_v1` 的代码能力：**普通候选只要被选Screen cell的总体净收益>0，且满足原有回归/stuck/hard风险检查，可以不修自己声明的target而进入Full-E与V；`v_primary`不再要求普通Full-E superiority作为晋升前提。它不是“E得分好就部署”，仍由V多对手配对Gate决定。

入口：[global_e_screen()](/Users/spring/RSI/Evotau-global-screen/src/evotau/evolution_gate.py:280)、[screen与后续路径](/Users/spring/RSI/Evotau-global-screen/src/evotau/skill_evolution.py:895)。这里“global”指当前Screen子集的总体收益，Screen仍是target/protected/prior-fixed/clean的子集，不是自动展开全E。

**Frozen Discovery Target 存在时：**[_screen_protocol()](/Users/spring/RSI/Evotau-global-screen/src/evotau/skill_evolution.py:31) 强制回`target_fix_v1`；933行再用专门target与preservation判断覆盖Screen；随后要对target及E重放保护和V全部过关。因此“没有定向修好，却一般任务变好”的候选仍不能算那个Frozen Discovery的Targeted Repair成功。这是当前冻结协议的明确限制，不是已经修复或验证了的一般能力。如果研究需要让这种候选作为独立general improvement晋升，必须另立协议，不能把它伪装成指定Discovery被修复。

最新global配置和离线测试没有产生真实V、A/A或Repair Pair证据；本场Pilot不运行Service生成，不能用它回答V晋升率。当前hard_policy_protocol_violations在native runner中没有业务合规检测赋值，默认0；虽然Gate会比较该字段，**“0 hard regressions”并不证明无严重违规**。混合text/tool另有独立观测，也没有自动映射成hard字段。[EpisodeRecord默认](/Users/spring/RSI/Evotau-analyst/src/evotau/records.py:42)、[runner建record](/Users/spring/RSI/Evotau-analyst/src/evotau/tau_episode_runner.py:478)、[Gate使用](/Users/spring/RSI/Evotau-global-screen/src/evotau/evolution_gate.py:80)。

## 6. 根因优先级表

| 问题 | 影响的研究假设 | 证据强度 | 离线可核对 | 最小验证 | 是否需改代码 |
| --- | --- | --- | --- | --- | --- |
| Policy发现与Native选择错位 | Customer发现真实弱点→保留有效挑战 | 高：Task39 gold/policy/DB/工具/评分全链，Task7二值折叠 | 是 | 对已有V2二十条候选建立独立行为scorecard | 若要在线保留该类发现，需要opt-in诊断；不改Native |
| 缺失DB/COMM组件 | Evolver从真实失败定位方向 | 高：真实input缺字段，Task7 DB正确 | 是 | 原请求与原reward_info一一对照；组件补充不能泄露gold | 最小反馈字段改动，有新配置身份 |
| 新颖性/暴露没有实测 | 一个新Skill→一个新挑战 | 高：Task15已存在；Task0无后续回合 | 是 | 给已保存轨迹标注original机制、trigger机会、actual exposure | 先诊断；需要机制记录时新版本 |
| Customer漏约束/扩授权/迟问 | 合法Customer→可解释Service差分 | 高事实，中等归责：Task12预算未披露，Task7迟问 | 是 | 盲审争议例及审核metadata边界 | 只修明确定义的Validator契约，不降低标准 |
| 普通rollout波动与全局prompt作用 | 观测差分→策略因果收益 | 对话变化高，收益归因低 | 只能部分 | 已有被动A/A；后续有授权再做随机化重复 | 不需要先改评分或Gate |
| Target Screen/真实V未触达 | E产生候选→真实Pair→Open-RC反馈 | 高：一Screen两语义拒绝、V=0 | 是 | 复核既有证据及global路径；真实V必须另外运行 | global已提供普通候选路径，不能宣称效果已解决 |

## 7. 最小修改建议（尚未实施运行代码）

### 建议1：独立 Behavioral Discovery / Policy Violation 诊断档案

**行为改变：**原生Native分数照常保存/选择；另存“场景忠实性、实际暴露、公开Policy/授权行为、证据充分性、相对已有E证据的新颖性”。即使Native0→0或0→1，也能保存待复核行为发现。没有Repair Pair时叫普通Behavioral Discovery，不能叫Repair-Conditioned Discovery，也不给Bandit正奖励冒充已复现发现。

**为何：**直接补上最确定的盲区，先回答到底有没有合法的新错误，再决定是否需要改研究fitness。不要把Task39错误gold替换成自造新reward，也不要把“policy检测未知”记录成“已合规”。

**精确入口：**在 `customer_evolution.py::evaluate_candidate()` 的原轨迹合法性审核后，追加独立sidecar；`customer_diagnostic.py::run_customer_diagnostic()` 无论是否selected都归档；不要复用需要promoted Pair的 `RepairDiscoveryArchive.publish()` 语义。先实现只读诊断，不更改Customer选择或Service Gate。

**风险/测试/版本：**盲审者不能看候选组名、Native结果或gold，否则confirmation bias；只用公开Policy、实际对话/工具，隐藏场景由独立Customer合法性审核处理。测试Task39非覆盖取消、Task7额外授权、合法升级、API拒绝而未扣款、证据不足、Pass/Fail四组合；先手工校验判定协议。任何新增在线模型审查或改变fitness/Gate都必须新算法/manifest/目录，不混入历史实验。

### 建议2：提供不含gold的评分组件与覆盖信息

**行为改变：**在 `alternating.py::_project_context_episodes()` 读取已保存native result，只暴露实际DB/COMM分值、终止与可用性；`evolution_context.py::build_customer_evidence()` 保留全E概览，并显式报告哪些失败组件/已有挑战被代表案例覆盖。不要把 `communicate_checks.info` 的预期答案、reference actions、最终gold DB或隐藏目标塞进Evolver。

**为何：**Task7“业务未完成”与“通信目标未命中”不该只有同一个失败标签；覆盖披露也能提醒Evolver这已是原生机场挑战。不是自动硬编码mutation taxonomy。

**风险/测试/版本：**组件依旧可能受错gold影响，不能自动称Service违规；测试未知值不补0、E全覆盖、ref/hash保真、无V/H/场景向Service泄漏、无target答案泄漏。输入契约变化必须冻结新配置，不复用旧Evolver缓存；原生episode是否复用另做compatibility audit。

### 建议3：把“预期trigger / 实际暴露 / 任务忠实性”拆开，修正审核对象边界

**行为改变：**保留开放式Customer机制，不增加固定操作者或强迫进化。对已执行程序先记录适用机会和真实用户行为，区分被原任务覆盖、没有机会、Service正确防守、真实错误、普通合作及uncertain。静态Customer Validator明确 `skill.evidence_refs/hypothesis` 是审核metadata，只审核compiled overlay的runtime泄漏；轨迹Validator明确预算、条件授权及要求的询问时点。不要因为Service违规就判Customer非法，也不要靠一句“valid”免掉所有争议。

**入口：**`customer_skills.py::CustomerSkill.compile()`、`customer_evolution.py::evaluate_candidate()/review_customer_runs()`、`evolution_candidates.py::CUSTOMER_VALIDATOR_PROMPT`、`customer_trajectory_validity.py::TRAJECTORY_VALIDATOR_PROMPT_V2`。

**风险/测试/版本：**开放trigger未必可被确定性规则完全判定，允许uncertain；正常价格询问/原有回退/知情授权不能算额外程序协助。测试审核metadata不误拒、metadata仍不进入Customer、Task7时点/第二项条件与Task12未披露预算、Task15已有机制、human transfer无暴露机会。Prompt或审核协议更改必须新实验版本；不修改正在运行/已冻结配置。

**下一步只做一件事：对已经完成的 V2 seed1 二十条候选轨迹做匿名化、逐消息的离线 Behavioral Discovery 盲审，Task39/Task7作为已核实锚点。**先判“合法Customer实际暴露了什么新行为错误”，同时保留Native矩阵。零新API，比再换模型或扩大搜索更能消除“没有弱点 vs 指标漏弱点”的最大不确定性。

## 8. 验证状态与复核方式

### 实际数据已经验证

- 四份冻结运行的94条完成轨迹、十个E任务，严格原生DB replay与原生COMM分数一致；原始文件完整未改。
- Task39参考取消与保险覆盖条件冲突；七条MSJ4OA取消复现得Native1。
- Task7 C1确实额外升级59XX6W，DB降分被总分零遮住；708与查询时点的剩余未来预订账单一致。
- V2实际生成请求缺评分组件；Task15机场机制已经存在于该请求的原始详细证据。
- 两候选20场prompt正确安装；既有十组baseline A/A同prompt仍产生不同对话。
- Pair Acquisition一例旧Screen拒绝、两例运行时实体泄漏的语义拒绝、V=0；Task12新运行未披露预算且六次不激活。

### 代码验证，尚不能当作真实效果

- latest global_e_v1普通候选可因非target修复进入V；Frozen Discovery仍走target协议。
- 原生Gate中hard字段没有全面Policy检测来源，不能把默认0当作安全认证。
- 独立Discovery Archive的存在和恢复能力不代表本Pilot运行了Repair-Conditioned发现链。

### 仍未验证

- V2是否稳定提高合法、新颖的行为发现率；其涨分/额外错误的因果归属。
- 程序性协助率、Validator误判率、A/A总体分数噪声率。
- 真实global V晋升、真实Repair Pair、Open-RC固定Pair对照和H泛化；本审计没有读取H帮助判断。

审计脚本没有构造Agent/UserSimulator/Orchestrator。它对socket connect设硬性禁止，只调用deterministic native tools/evaluators；原环境重放会在内存模拟写工具，**不改原DB文件，不是线上episode**。分析输出只写当前审计目录。

复核命令（只读，无模型请求）：

```bash
cd /Users/spring/RSI/Evotau-global-screen
/Users/spring/RSI/Evotau/.venv/bin/python research/audits/customer-evolution-root-cause-full-20261011/analyze.py
/Users/spring/RSI/Evotau/.venv/bin/python research/audits/customer-evolution-root-cause-full-20261011/summarize.py
/Users/spring/RSI/Evotau/.venv/bin/ruff check research/audits/customer-evolution-root-cause-full-20261011
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/pytest -q tests/test_global_e_screen.py
```

本轮实际测试还包括 `tests/test_customer_prompt_v2.py`、`tests/test_customer_native_indices.py`、`tests/test_evolution_context.py`，四个模块共60项；通过Python入口给socket连接设禁止后执行pytest，详情见 `validation.json`。唯一warning为原生依赖中Python的audioop弃用提醒，与研究判断无关。

文件索引：

- `inventory.json`：94场逐条身份、分解分数、实际写操作、回放后的reservations差异、原始SHA；50组Customer配对与首次文本分叉。
- `findings.json`：明确扫描分母、判定范围、named-rule证据、零基原消息索引/行号/原文、已有baseline A/A及Task7账单。
- `E-transcripts.md`：同批94条E对话的只读摘录，保留所有用户/助手发言与工具调用、WRITE结果；没有把完整READ catalog复制进本报告。
- `E-only-task-evaluation.json`：只解码的十个E原任务副本，含离线审计需要的评分定义，**不供Evolver读取**。
- `analyze.py` / `summarize.py`：可复现只读统计/确定性环境复核；不是新研究运行机制。
- `validation.json`：本轮实际测试命令、退出状态、固定源码校验与输入完整性核对。

主实验的HTTP402只解释A/B为何没有全四组最终结果，不解释seed1挑战的科研效果；没有根据余额不足替换模型或恢复收费动作。本报告不把部分seed2候选的已保存分数当作已审核的合法发现。
