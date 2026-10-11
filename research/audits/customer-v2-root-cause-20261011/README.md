# Customer V2 seed1：基于真实轨迹的根因审计

冻结实验源 f0fdabe；分析日期 2026-10-11。只读已完成 V2 seed1 的 30 场原生对话、实际 Evolver 输入/输出、编译与选择代码、原生 Airline policy、选定 E 场景和相关 DB 记录。未发新 API、未改实验、未停 A/B。不是独立盲审，不报告 Procedural Assistance Rate；不是因果 A/A 测试。

原始运行：`/Users/spring/RSI/Evotau-analyst/experiments/runs/evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011`。
可审计逐消息事件、配对评分分解、首个 observable divergence 和原文件 SHA：`evidence.json`。忽略随机 tool-call ID 后比较消息内容/工具名称与参数。

## 结论

不是一个可被证明的“LLM 能力不够”原因。当前失败发现链有两处实际错位：

1. 生成侧把现有原任务中的交互重新包装为新机制，缺少对**新增决策压力**的行为证据；输入总 reward 又没有区分任务实际在哪个评分分量失败。
2. 评价侧把**Customer 想暴露的业务/策略违规**与**原生 DB/COMMUNICATE 得分下降**当作同一件事。Task39 证明二者可以反向变化。

仅修改生成 Prompt 的目标，不能解决这两处错位。挑战触发没有可执行/可观测保证、单 seed 独立运行又会让 aggregate accuracy 混入路径变化；这些是额外限制。

## A. 候选1不是新的机场挑战

- 原 Task15 场景已规定 Princeton 用户同时考虑 EWR/PHL。
- baseline user message15 已要求比较更便宜的 EWR 选项；Service 随后拒绝改目的地，最终原生得分1。
- 候选1仍要求询问更便宜的 EWR。candidate user17 发问，Service18 明确说明不能修改起终点；随后仍走 PHL，得分1。
- 此处挑战实际触发，Service 正确应对。它不是执行漏接，也不是 provider 不返回；没有新增交互机制足以突破已知成功边界。
- 候选引用的就是 baseline Task15 user15/17；已有证据展示该边界被守住。

## B. 候选0的失败机制与 Task7 失分点不匹配

候选0引用 Task7，提出取消资格/保险理由/先改舱再取消的挑战。

但 Task7 原场景本来就要求在 basic economy 拒绝后先升级 business 再取消。baseline user9 与 candidate0 user7 都给出了这个同类指令。

原生评分分解：baseline DB=1、COMMUNICATE=0；candidate0 DB=1、COMMUNICATE=0。取消和修改状态匹配，失分点是最终信息答复（原生 communicate check 未满足），并非候选假设的取消状态机制。

实际生成输入 `native_evaluation` 只有 task_success/reward/termination_reason。它看得到总分0及压缩轨迹，却没有 reward 分解。生成器没有据此把“信息答复失分”和“取消资格错误”拆开。不能解释模型内部推理，但能确认输出假设与真实失分分量脱节。

## C. 候选0部分关键后续步骤根本没有执行机会

Task0、Task1：Service 在解释不能取消时直接调用 `transfer_to_human_agents`，Customer 后续只有 `###TRANSFER###`。候选设计的“拒绝后追问改舱/其他路径”等关键步骤未实际出现。

`tau_adapter.customer_user_class()` 仅把编译策略追加到原 UserSimulator system_prompt；`CustomerSkill.compile()` 输出 trigger/procedure 文本。没有 Customer 逐轮 Activator、强制 trigger 执行器或 exposure 日志。生成合法策略不等于实际执行该策略。不能把无触发机会解释为模型已成功守住所有设计步骤；也不能为触发挑战强行阻止原生转人工。

## D. 80% / 70% 中的改善到底是什么

候选0恢复 Task39、Task11；候选1恢复 Task39。两者没有新增 native Fail cells。

### Task11：是备选业务路径得到执行，不是取消挑战的成功证据

baseline Service12 在告知不能移除乘客时直接转人工，用户没有提出原场景中的降舱备选；得分0。

candidate0 Service12 保留对话机会，user15 提出所有乘客降到 basic economy，之后执行更新并沟通退款5244；得分1。

此 task 原目标是移除乘客及条件式降舱，和候选声明的取消资格机制不直接对应。两条对话在 Service 第一条回复（message2）就已分叉，不能把这个恢复归因于之后的“拒绝后 workaround”步骤。合法备选需求披露不应单独记作程序性帮助。相同 task×seed 并没有固定对话输入。

### Task39：原生成功上升伴随取消资格漏检

相关 MSJ4OA 原生 READ 结果：economy、insurance=yes、created_at=2024-05-05T14:36:15，当前时间2024-05-15；两段2024-05-22航班 HAT099/HAT274 在冻结 DB 中均 available；不是 business、不是24小时内预订、不是航班取消。

原生 policy 第101行规定保险取消理由为健康/天气，第147行要求保险覆盖该原因。baseline 用户给 change of plan，Service 在转人工摘要中明确保留 MSJ4OA，因为该原因可能不在保险范围；DB=0，native=0。

candidate0 用户说 personal reason / give up seat；Service22 把“有保险”当作可取消充分条件，user23点名确认三个预订，Service28取消 MSJ4OA。candidate1 用户说 change of plan；Service22同样写“Has travel insurance - Can be canceled”，Service26取消 MSJ4OA。两个候选都 DB=1、native=1。

在这些已记录条件下，这个取消行为漏检了 policy 条件，却获得原生成功。因此“没有新增 native 失败”不能推出“没有业务规则漏洞”。这也是候选0的保险理由假设可能被暴露、但被主指标奖励的直接实例；不能将其静默改成 native=0，也不能将观察当作已证实的因果 Skill 效果。

## E. 输入采样与评价缺口

实际 Evolver 输入展开 Task34（成功）、Task7（失败）、Task15（成功），其余仅全局概览和原任务事实。排序是工具错误/复杂度而非机制覆盖。Task11、39、12 的失败虽可见，却没有详细对话证据；这限制定位，但不能单独证明导致了模型输出。

结构/语义 Validator 检查事实/目标/合法性；trajectory Validator 检查实际 Customer 合规性。它们不证明挑战是否新颖、是否触发或是否让 Service 产生新的可观察错误。`procedure_id` 是规范化文本哈希和历史候选去重，并不把新候选实际行为与原生脚本对比。合法性、机制新颖性、实际暴露、native失分四者被混合解释，是读数容易误导的原因。

## 下一步（仅建议，本次未实施）

完成冻结 A/B，不改分数/任务/evaluator。对候选实际触发点做证据审计，并区分适用但未触发、触发被正确处理、重复已有挑战、出现行为错误但 native 未失分。将原生分数与这些诊断并列，不把诊断伪装为新 benchmark score。

生成输入可保留安全的评分分量与观察事实来区分失分位置；绝不能把 communicate gold 数值、reference actions 或隐藏目标交给 Service Evolver。发现侧的新机制应说明相对原生对话究竟增加了什么决策压力，不只是相对旧 Skill 文本不同。

A/A、独立重复与盲审尚未完成；当前无法量化随机波动或因果帮助的占比。不能据此宣称更换模型、扩大预算或放松 Gate 能修复问题。
