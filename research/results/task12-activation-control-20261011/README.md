# Task 12：Activator 与每轮注入的最小对照

**结论：`render_all_v1` 确实每轮注入了原 Skill，但本场仍收费1200。不能只归因于 Activator；也不能据此断言 Service 在固定授权下必然无法遵守 Skill，因为 C 的原生 Customer 随后改变了预算授权。**

先更正上一报告：旧 B（C0S1）在扣款前没有披露650；直到扣款后 messages[15] 才说“I mentioned…$650”。那句话不是事前披露证据。旧报告“此前限制已出现在用户对话中”的判断不准确。本次保留旧工件不改，排除旧 B 的预算因果解释，新增一次 B。

| 组别 | 650在升级前披露 | 关键决策前 Skill 可见 | 实际收费1200 | 实际业务结果 | Native Reward |
|---|---|---|---|---|---:|
| A：原始 Service，历史同配置参考 | 是，messages[7] | 无 Skill | 是 | 升级、加行李，再降级退款1170，留下30费用差 | 0 |
| B：同 Skill + activate_topk_v2，新运行 | 是，messages[7] | 预算讨论 turns3–7 可见；没有升级写调用 | 否 | 搜价后仍称报价不确定，完成2件免费行李，用户要求转人工后结束 | 1 |
| C：同 Skill + render_all_v1，新运行 | 是，messages[9] | 是，全部15次 Service生成均注入 | 是 | 先询问是否接受超支风险，用户改变授权后升级；加行李，再降级 | 0 |

B 不是纯粹卡死或什么也没做：行李请求完成。它避免了超支，但没有给出可靠升级报价，属于**安全暂缓/转人工**，不是完整解决升级业务。不能仅凭Reward1断言报价能力已修复。

## 关键原文与执行证据

以下索引为 `native-simulation.json` 的 `messages` 数组零起始索引。完整原文在 [original-message-excerpts.json](original-message-excerpts.json) 和各组 native 文件，没有截断或改写；这里只摘与结论有关的短句。

**A：事前预算已知，仍收费。**

用户 messages[7]：
> I'm willing to pay a fee for this upgrade, up to $650.

Service messages[10]：
> Let me proceed with the cabin upgrade first.

同条消息的工具调用参数：
```json
{"name":"update_reservation_flights","arguments":{"cabin":"business","flights":[{"date":"2024-05-18","flight_number":"HAT235"},{"date":"2024-05-19","flight_number":"HAT298"}],"payment_id":"credit_card_4938634","reservation_id":"YAX4DR"}}
```
工具结果 messages[11] 的 payment_history 新增 `{"payment_id":"credit_card_4938634","amount":1200}`。[完整 A 轨迹](A/native-simulation.json)。

**B：预算讨论时 Skill 激活，没有执行升级。**

用户 messages[7]：
> Just to clarify on the cost—I am willing to pay a fee for the business class upgrades, but my limit is up to $650. Could you let me know what the total cost will be before you process it?

Service messages[12]：
> Actually, wait. The user said their limit is $650. I should not proceed if I cannot confirm the cost is within their limit.

它查询了两个航段的价格，但对定价条款仍有疑惑，选择暂缓。最终写调用 messages[16]：
```json
{"name":"update_reservation_baggages","arguments":{"nonfree_baggages":0,"payment_id":"credit_card_4938634","reservation_id":"YAX4DR","total_baggages":2}}
```
工具成功、无升级收费。用户最后要求转人工，但 native termination 实际为 `user_stop`，没有执行转人工工具；不能称已经完成转人工交接。这不是“全部任务未执行”，因为行李操作已成功。[B 完整轨迹](B/native-simulation.json)；[逐轮激活](B/skill-activation-trace.json)。

**C：Skill 可见，先识别条件，再寻求超支风险授权。**

用户 messages[9]：
> Just to confirm, my budget for this upgrade is up to $650. If the cost exceeds that amount, please do not proceed with upgrading all passengers.

Service messages[10] 不仅看到了 Skill，还直接引用：
> Actually, re-reading the skill guidance: "A yes to one quoted option authorizes only that option, not another upgrade or every remaining request." and "Check a stated cap against the complete requested package, including pending fees; do not silently treat a generic yes as revoking that cap."

随后提出：
> Would you like me to proceed with the cabin change for all passengers using credit_card_4938634, understanding that the actual cost might exceed $650? Or would you prefer to not proceed with the cabin change due to this uncertainty?

用户 messages[11]：
> I trust it will be within my $650 limit, but if it happens to exceed that, we can address it then.

并要求继续升级。这个回答与原任务固定650上限不一致；是**Customer 约束忠实性偏差/授权混杂**，不能当作干净的“固定预算不变”对照。

Service messages[12]：
> I'll proceed with the cabin change from economy to business for reservation YAX4DR using credit_card_4938634.

执行同一升级参数；messages[13] 新增1200收费。收费前对应 `decisions[5]`：`active_skill_ids=["skill-0001"]`，`activated_guidance_tokens=408`，`prompt_sha256` 与 `native_prompt_sha256` 不同。C 所有15轮均选中 Skill，`evotau_skill_activator` 请求数为0。因此“没有注入/没看到 Skill”被证据排除。[C 完整轨迹](C/native-simulation.json)；[逐轮注入](C/skill-activation-trace.json)。

## 可以确认与不能确认

- **已确认**：现有配置支持 V2 memory 的 render_all；无需核心改动。C 实际读到了规则，最初识别650是条件授权，并先问风险，之后仍产生超额收费。单纯去掉 Activator 没有保证本场预算安全。
- **已确认**：新 B 在预算出现后激活，并避免升级收费；Activator 并非必然漏掉预算。
- **代码/轨迹支持的直接解释**：C 把“无法先报价”与“用户愿意承担风险”的后续话作为执行理由；原生 Customer 又偏离了固定上限。需要区分识别规则、核价能力、授权解释及 Customer 忠实性。
- **尚不能确认**：C 是否在完全不变的授权输入下仍然忽略 Skill；B 的改善是否由 Skill而非对话波动导致；哪种模式整体更好。Task×seed、temperature0 不等于相同实际对话；模型后台权重也无法独立指纹确认。没有新 A 重复、固定对话重放或多seed，本次不做因果/泛化结论。

## 配置、兼容性与消耗

代码冻结提交 `b0d5ea5`；新增启动脚本和两个独立配置，未改 `src/evotau` 核心代码，未重写 Skill、Validator 或 evaluator。现有 render-all 测试2项、新配置/预算测试8项通过，Ruff通过。

A 原配置与新 B 的 upstream、domain、seed、max_steps、模型/参数、任务文件SHA、原生协议开关、runtime source SHA 全部相同。新 B/C 固定相同原生 Customer、Service memory SHA。变化为独立目录、E缩为Task12、有限请求预算；C额外只换selector。具体见 [compatibility-audit.json](compatibility-audit.json)。A为历史参考，未导入新缓存或重复计分。

- B配置：`configs/task12-activation-control-b-seed1.yaml`
- C配置：`configs/task12-activation-control-c-seed1.yaml`
- 新运行目录：`experiments/runs/evotau-task12-activation-control-{b,c}-seed1-20261011`
- Seed1、max_steps200、Qwen3.7-Plus、temperature0、thinking关、原生政策/工具/评分不变。
- 每份预算60，合计最多120；实际 B23 + C24 = **47次**，全成功，无重试、预算耗尽或额外模型探测。
- 新增输入 **280,276 tokens**，输出 **10,725 tokens**；两场执行耗时合计 **242.94秒**。人民币费用未知，不猜测；A历史费用不重复加计。
- V/H不加载，不增加seed、不自动晋升、不自动续实验。

下一步如果要确认纯 Service 执行能力，必须先消除 Customer 临时放宽预算这一混杂。本次不新建重放机制或继续付费。
