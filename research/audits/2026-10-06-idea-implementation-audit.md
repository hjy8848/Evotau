# EvoTau IDEA 与实现审查 · 2026-10-06

审查分支：`codex/alternating-evolution-refactor`。审查基准：`9c574c19266631390ce258d20acef31ab36c2624`。

本次检查最新 SkillMemory 需求、配置、执行代码、已保存的真实实验、原生 τ-bench 源码，并做离线复现。未修改算法，未调用真实 provider，未启动实验。`ruff check src tests` 通过；完整测试为 **92 passed in 4.37s**。离线补查发现了现有测试未覆盖的问题。

## 结论

IDEA 有清楚、可检验的核心：固定任务、环境和执行模型，Customer 搜索更难的交互方式；Service 从这些真实交互中学习，通过外部 SkillMemory 改变行为。当前实现已经具备这个闭环的主要组件。

当前首先需要解决的是实验正确性与运行可靠性。尤其是 **Flash runtime 的“关闭思考”配置并未按已有转换规则送到 provider**。真实保存的回复证实 runtime 实际产生了 reasoning tokens。这会同时影响模型对照、耗时和对失败的解释。

当前证据还不能证明持续 Service evolution 有效，也不能证明 IDEA 失败。最近一次 SkillMemory run 在最初的 incumbent panel 就中止，尚未调用任何 Evolver。

## 1. 当前 IDEA 的准确版本

1. τ-bench 的 Retail tasks、原始 user scenario、policy、backend、tools 和 native evaluator 固定。
2. Customer 使用开放的 `PromptStrategy`，每代提出一个候选；在固定 E20、fitness seed=1 下，严格更低的 Service accuracy 才替换 incumbent，tie 保留 incumbent。
3. Service runtime 固定为 DeepSeek V4 Flash。强 Evolver 是 DeepSeek V4 Pro；其生成的是外部技能，而不是模型权重更新。
4. Service 从空 `ServiceSkillMemory` 开始；每代最多一个 `ADD`、`UPDATE` 或 `NO_OP`，所有 active skills 都注入原生 Service prompt，无 retrieval/selector。
5. 当前是 E-only mechanism smoke：Service proposal 严格提高相同 Customer 条件下的 E accuracy 才接受；V/H 暂时关闭。不是 generalization 或 forgetting 实验。
6. 不使用 Reviewer、Customer Judge 或 Service Judge。仍使用 native evaluator，包括任务需要时的 LLM NL assertion evaluation。

这可以验证“交互压力 → 局部行为经验 → 相同挑战下的准确率改善”。它尚不能验证长期稳健性、跨任务泛化，或比整体 PromptStrategy rewrite 更少 regression。

## 2. 实现核对

| 项目 | 当前情况 | 证据 |
|---|---|---|
| Customer/Service/generation 顺序 | 符合，phase 之间没有交叉并行 | `alternating.py:481–1102` |
| Customer strict-lower / tie incumbent | 符合 | `alternating.py:592–600` |
| Service strict E improvement | 符合；V 开启时另加 aggregate V 不下降 | `alternating.py:819–879` |
| empty SkillMemory / ADD / UPDATE / NO_OP | 已实现；immutable 数据和稳定 ID | `service_skills.py` |
| rejected ADD ID / resume | 有 high-watermark 和恢复测试 | `alternating.py:748–752,1081–1100`；SkillMemory tests |
| inject all / policy precedence / provenance 隔离 | 已实现；empty memory 不加模板 | `service_skills.py:288–320` |
| Service 不直接收到 `user_scenario` | 已结构隔离；history 仅数值，accepted Service history 只来自 Service | `alternating.py:610–631,1066–1074,1631–1690` |
| gold/reference evaluator target | context 投影不包含它们 | `alternating.py:1651–1715` |
| E20 / V / H / excluded | 实际 split 检查通过：20 个唯一 train IDs，互不重叠，无 46/47 | 见 evidence JSON |
| H sealing | 当前 E-only CLI 不构造 H task objects，不向 Evolver 提供 H 内容 | `alternating_run.py:75–94,190–212,399–581` |
| RequestBudget/concurrency | 有锁、线程局部 episode accounting、reservation 和 ordered results tests；当前并发 1 | `budget.py`；`alternating.py:1537–1621` |
| Reviewer/Judges | active alternating path 为 0，真实完成 run 的 usage 也为 0 | `tau_episode_runner.py:304–310`；旧 Flash result |
| Flash runtime model args | **不符合已有 provider 转换规则；见 F1** | `tau_episode_runner.py:77` |
| 缺失 native score | **会污染 accuracy；见 F2** | `alternating.py:1718–1721` |
| resume identity | **提交结果后就可能无法续跑；见 F3** | `alternating_run.py:63–65` |

Service-side Retail task descriptions 在检查的 E20 archived context 中均为空；当前没有从该字段发现额外私有事实。开放 Customer strategy 的语义仍依赖 prompt：不加载隐藏场景字段，并不自动证明每条生成策略都没有复制事实或改变目标。

## 3. 可复现的工程发现

### F1 · P1：episode runtime 没有转换 `thinking_mode`

`role_model_args_for_runtime()` 已经把配置中的 `thinking_mode` 转成 `extra_body.thinking.type`。CLI 对 Evolver 使用了这个函数，但 `TauBenchEpisodeRunner.__init__()` 直接从 manifest 复制原始 args。Agent、Customer、Evaluator 随后收到的是原始 `thinking_mode`。

使用真实 LiteLLM + OpenAI SDK 序列化、`httpx.MockTransport` 拦截请求（没有网络请求）得到：

```json
当前 episode 路径：{"temperature":0.0,"thinking_mode":"disabled"}
已有转换规则：     {"temperature":0.0,"thinking":{"type":"disabled"}}
```

这不仅是静态推测。最新 run 的两个完整 episodes 的保存响应中：

| role | calls | 有 reasoning tokens 的 calls | reasoning tokens |
|---|---:|---:|---:|
| Customer | 14 | 14 | 1,596 |
| Service | 21 | 20 | 2,409 |
| 合计 | 35 | 34 | 4,005 |

这 35 次调用共报告 6,624 completion tokens，其中 4,005 是 reasoning tokens。配置明明要求 runtime `thinking_mode: disabled`，这个控制变量没有实现。不能用这些数据声称已经测过“Flash 不思考 runtime + Pro 思考 Evolver”。

最小修正：让 episode runtime 复用已有 args converter；测试真实 runner → τ generate → LiteLLM 请求边界，而不是只测试 converter 的返回值。修改后需新 manifest，并验证真正送出的参数与返回的 reasoning metadata。

**这不证明 task 29 的空回复就是由该错误引起。失败回复未保存，直接因果仍未知。**

### F2 · P1：未知评分被当成 task failure

episode runner 在 native reward 缺失或非有限数字时产生 `EpisodeStatus.UNCERTAIN`、`task_success=None`，并保存 record。但 `_accuracy()` 仅计算 `task_success is True` 的占比，因此 None 自动进入分母并记为 0。

离线复现：一个 UNCERTAIN record 的 `_accuracy([record]) == 0.0`。

这会让没有有效评分的 Customer candidate 看起来更难，也会让评分缺失被错误解释成 Service 退化。SkillMemory 的 paired diagnostic 会检查 None，但发生在后续 Service 阶段，不能保护前面的 Customer selection。

最小修正：计算 fitness 前要求完整、有效的 native boolean score；未知评分保存为 incomplete/unscored，阻止选择，不能擅自计失败或删出分母。检查时也不要把缺失轨迹默认为一条有效空对话。

检查的真实完整 Flash/GPT panels 暂未发现 UNCERTAIN records；这是离线复现的潜在选择错误，不是本轮 task 29 的已证实原因。

### F3 · P1：provenance 与 resume compatibility 被绑成一个完整 manifest 比较

`AlternatingManifest.from_mapping()` 每次读取当前 Git commit、dirty 状态和源码 hash；CLI 要求现存 manifest 与重建文档完全相同。

最新 run 的实际对比：

```text
saved commit: 340c78f91700a782574fecf9bfdcc433afa4baf8
current commit: 9c574c19266631390ce258d20acef31ab36c2624
两者源码 SHA256: 6f8b34145687b1294075751b076c44a90c5ec0bddf55751d4c21827b6bda4380
源码相同，但重建 manifest 不相同。
```

因此，单纯归档并 commit/push 运行结果，也会令原配置无法经普通 CLI resume。dirty 状态变化也有同类问题。另外，source fingerprint 纳入所有 configs，而非仅当前选中的配置，所以新增无关实验配置也会改变旧 run 的兼容性。

最小修正：保留原始 provenance；resume eligibility 校验实际执行源码、选中配置、冻结模型/args、benchmark pins 等实质输入，commit/dirty 状态作为每次 invocation 的审计信息。不能因此允许真正改变 runtime 条件后无条件复用旧 scores。

### F4 · P2：失败留下的证据不足，runner 状态也不会完整收尾

当前空回复得到 API-level success accounting，之后在 `UserMessage.validate()` 抛错。episode runner 保存 exception、usage 和 prompt hashes，然后再次抛错；`_run_panels()` 中止 panel。完整 panel 是计算 fitness 的必要条件，这个约束合理；问题在于失败任务缺少可诊断和可恢复的记录。

`on_simulation` 只在整个 simulation/evaluation 返回后执行。中途失败时没有保存部分 conversation、最后一轮原始 response 的诊断 metadata、finish_reason、reasoning/visible-content 状态。CLI 只打印失败并返回 2，`run-execution-state.json` 的 status 也只在成功路径更新；额外 stop summary 不是通用失败处理保证。

最小修正：持久保存每次请求的脱敏 metadata，至少包括 role/model、响应 ID、finish_reason、visible/reasoning/tool 状态、usage、latency 和异常；保存 partial trajectory；错误路径写入明确失败阶段。允许 operator 对未完成项进行显式 resume。不要把空回复伪造为 STOP，不要把 reasoning 当作用户对话，不要跳过任务后继续选择。

此前“不是 token 上限”的表达需要收窄：已排除的是 EvoTau request cap 和配置中的显式 `max_tokens`。**没有 finish_reason 等失败响应证据，不能排除服务端默认输出截断。**

### F5 · P2：MultiToolMessage 的 tool results 被 context 投影丢弃

`_trajectory_context()` 只保留 `role/content/name/tool_calls/tool_call_id`，不保留 `multi_tool.tool_messages`。离线输入含工具结果的 MultiToolMessage，输出只剩 `{"role":"multi_tool"}`。

这会在原生 runtime 产生该消息类型时让 Evolver 缺失工具结果。检查的两组 archived Flash/GPT trajectories 没有 MultiToolMessage，所以当前没有证据说明本轮失败由它导致。

最小修正：结构化展开或保留嵌套 tool messages，并测试投影。无需增加研究机制。

### F6 · P2：同一条件会因 panel_name 不同重复执行

episode cache key 包含 `panel_name`。generation N 的 accepted pair 在 generation N+1 的 incumbent panel 中，即便 task、Customer、Service 和 fitness seed 完全相同，也不能直接命中已有 episode。

这既浪费时间，也使所谓“固定 fitness seed”仍有重新采样：原生 τ-bench 会向 agent/user 传 seed，但 provider 是否确定性并未验证；temperature=0 也不能自行证明每次回复相同。此点不是当前停止原因，但会影响 accuracy 曲线解释。

建议明确区分条件 identity 与引用这些结果的 panel provenance。相同条件可复用；独立复测则应显式标记重复测量。不要暗中改变 selection rule。

### F7 · P2：研究说明与当前方法已经不一致

`README.md` 仍要求冻结五个 roles（含 reviewer），描述 trajectory judges 和旧 V gate；`research/Alternating_EvoTau.md` 仍包含 Customer Judge、same-challenge judge 和 reviewer evidence。当前代码已经是四个 roles、native accuracy selection，并新增 SkillMemory。

旧说明不能再充当当前实验协议。本文按最新用户需求和代码审查，不沿用旧文档中的 Judge/Reviewer 设计。

## 4. 实际研究证据

| run | 已经发生什么 | 不能得出的结论 |
|---|---|---|
| Flash 5E/3V/5H，20261005 resume-safe | Gen0 Customer 1.00→0.80；Service 0.80→0.60 rejected。Gen1 incumbent 1.00；Service 1.00→0.80 rejected。50 完整 episodes，945 calls，4,557,933 prompt / 303,558 completion tokens，首次启动至完成 1,973.42 秒≈32.9 分钟；S0=ST，native/fresh H 均 1.00 | 没有接受任何 Service update。Gen0 fitness seed=1、Gen1=2，不能把这条历史曲线解释为当前固定 seed 机制；且 F1 会影响 runtime reasoning 控制 |
| GPT Evolver E20，20261006 partial | C0/S0 16/20；candidate 17/20，保留 C0；40 完整 episodes。Service Evolver 两次 client timeout、一次 HTTP 504，没有 Service proposal；记录的跨尝试时间约93.8分钟 | 没有 Service repair evidence，长上下文的 provider 失败不能归因于 alternating 算法 |
| V4 Pro long-context capability probe | 109,176 prompt tokens；移除 probe 的4,096-token输出上限后73.423秒返回合法 NO_OP JSON；8,006 completion tokens（7,770 reasoning）。仅一次未截断的大请求成功 | 证明可完成一次请求，不证明稳定 synthesis 或机制有效。Stage A 输出没有作为正式 skill 使用 |
| 最新 SkillMemory E20 | 两个完整任务66/92；task29两次空可见回复。51 calls，224,816 prompt / 12,441 completion tokens；Customer/Service Evolver均0 | 尚未测试任何正式 ADD/UPDATE/NO_OP、Service acceptance 或两代 SkillMemory evolution |

另一个长上下文质量信号：V4 Pro probe 的可见 `analysis` 把 archived E20 写成“14/17成功”，但实际输入是20个任务、16个成功，并明确给了accuracy=0.8。JSON合法、HTTP成功并不代表它准确理解了全部证据。这是 capability probe 的输出质量限制，不能作为正式机制失败或成功。

时间规模也需要按 episode 计算：当前 E20、每代一个 Customer candidate，每代要跑 incumbent 20 + candidate 20；Service proposal 改变时再跑20。两代最多120个 E episodes，即使连续 NO_OP 也仍有80个。在当前单 worker 下，每个 episode 内又有多次顺序 Customer/Service 请求，所以并不是40次单独 API 调用；长运行本身不能证明死锁。F1 的多余 reasoning 会额外影响这个规模的耗时。

## 5. IDEA 本身需要怎样解读

- **低 accuracy 不自动等于合理的 Customer challenge。** native reward 会反映 Service 失误，也可能反映用户未披露必需信息、目标漂移、确认与结束信号冲突或 max_steps 耗尽。当前 prompt-first 方案有意不加 Judge/黑名单；应对实际 pass→fail trajectory 做人工核查，而不是把所有降分都称为能力边界。
- **Skill mutation 局部，不代表行为影响局部。** 所有技能注入同一个 prompt，仍可能打坏其他任务。当前记录 paired task outcomes 有用，但 superiority 需要相同模型、panel、seed、runtime args 的 global rewrite 对照。目前还没有这个成功对照。
- **单个候选、单个 seed、两代可以做机制观察，不能证明稳定改善。** E20 的一个任务就是5个百分点；provider-level randomness 尚未测量。固定 seed 减少变化来源，不等于消除变化。
- **changing Customer 使跨代 accuracy 不严格可比。** 必须写清 C×S 条件。关键观察是同一个选定 Customer 下，S_old→S_proposed 的 paired delta；generalization 另看 native/fresh H。
- **NO_OP 或 rejection 是合法结果。** 没有更新可能表示证据不足、抽象质量不够、Service已接近该panel上限，或测量噪声；不能自动等同于实现坏了。反过来，一个 API failure 也没有给出 skill 学习是否有效的证据。

## 6. 下一步最小顺序

1. 修 F1、F2、F3，再补 F4 的错误证据和状态收尾；补真实调用边界的离线测试。继续保持目前的模型、prompt、selection 和 SkillMemory 机制。
2. 做 Flash runtime 的小型真实调用检查，核对实际关闭思考参数、visible output、tool calls、finish_reason 和 reasoning usage；再完成此前失败的 task29原生episode。
3. 条件正确后，按原定固定 E20、两代配置运行完整闭环。由于 runtime args 纠正改变实际实验条件，须使用新 manifest/output；旧结果保留，不能把旧 cached scores 假装成修正后 runtime 的结果。
4. 先报告 pressure、repair、paired regression 和 accepted memory；若只完成 smoke，就只报告 smoke。之后再决定是否进行 matched global-rewrite 对照或 V/H，当前审查不启动它们。

本次未改变任何研究机制，也没有真实 API 消费。离线证据及具体数值见同目录 `2026-10-06-idea-implementation-evidence.json`。
