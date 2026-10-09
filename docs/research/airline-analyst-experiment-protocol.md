# 新版 Airline 实验协议（待审阅，不自动启动）

日期：2026-10-09。代码基线：`007fed9aebbdc9703a6263378f245019fdc1ba84`。
算法：`analyst_skill_v_validation_v3`。
当前旧版收费实验继续在原 checkout 运行；本协议不修改它的代码、配置、结果或进程。
本次只准备协议，没有执行新的模型请求。

## 研究问题及可以得出的结论

首要问题：同样的 E 证据、模型和 Skill 表达，分析与生成分离后，是否提出更多有证据支撑、
具有不同操作机制的修复，而非三个同义提醒？
第二个问题：这些候选经过原有真实评测，能否在 V 晋升，并改善冻结后的 H scorecard？

这是一轮机制研究，不保证每代晋升。零假设、NO_OP、拒绝、INCONCLUSIVE、S0=ST 都是有效结果。
一轮 G2 的收益不能证明普遍有效；单个成功反转不能证明 Skill 的因果作用。
新版也调整了 Skill Validator 对“领域内可复用”的表述，并增加机制去重。因此完整运行比较的是
这套版本化生成流程，不可把改善全部归因于 Analyst 本身。若未来要隔离单个因素，需另行预注册消融。

## 冻结项

| 项目 | 本轮约定 |
|---|---|
| 领域和数据 | pinned τ-bench Airline，upstream b7ea9074，原生任务、policy、tools、backend、evaluator |
| Native roles | Agent / Customer / Evaluator / Activator：gateway `openai/dashscope/qwen3.7-plus` |
| Native args | temperature=0，enable_thinking=false，原 gateway 地址 |
| Evolver | 官方 `openai/deepseek-flash`，https://api.deepseek.com/v1，thinking enabled，reasoning_effort=high |
| 搜索 | G2，每代 Customer candidate=1；最多3个不同假设各生成1个 Skill；最多1个原有条件性 crossover |
| 执行 | episode 并发2；单场 turns 串行；max_steps=200 |
| 生成证据 | 当前 E 单 seed=1；全部结果概览＋既有确定性代表证据；不新增多 seed 诊断 |
| 初筛 | screen seeds=1,2；原有 clean-task 选择和所有阈值 |
| Full-E / V / H | seeds=1,2,3,4；原有配对和复用条件 |
| 统计 | 原 task-block bootstrap、Bonferroni、V-primary 接受规则不变 |
| 消耗 | request_budget_cap=null；不自行新增请求、输出 token 或上下文长度上限 |

E10：`1, 0, 39, 49, 11, 27, 34, 12, 7, 15`。

V20：`41, 20, 17, 40, 23, 33, 4, 10, 47, 28, 36, 46, 3, 14, 5, 21, 43, 9, 42, 38`。

H20：`2, 6, 8, 13, 16, 18, 19, 22, 24, 25, 26, 29, 30, 31, 32, 35, 37, 44, 45, 48`。

配置：`configs/airline-failure-analyst-qwen37plus-official-dsflash-p2.yaml`。
模型名、temperature、证据设置和所有阈值以该冻结配置为准，不为看起来更好的结果换任务或改参数。
A/A calibration 尚未完成；保持 calibration_confirmed=false。

## 步骤 A：相同证据的生成对照（推荐最先执行）

使用已保存 Gen0 E10 输入，不重新运行原生 episode，不加载 V/H。
旧组：同一证据下原有三个 bias Direct Mutator。
新组：同一证据 → 1次 Analyst → 有多假设时1次语义去重 → 0–3次 assigned Mutator。
两个组都记录所有原始输出、合法拒绝与 NO_OP；不能反复抽样直到生成满意的 Skill。
不额外调用 LLM 为这些输出打分。

费用规模是旧组3次请求，新组1–5次请求，合计4–8次 Evolver 请求。
这是固定搜索设计产生的调用范围，不是新加的费用上限。
原模型输入、API args 和 E-context digest 冻结在 replay bundle 中。
输出质量人工审核可用匿名候选编号，报告逐条依据，而不是只看写作质量：

| 指标 | 记录方式 |
|---|---|
| 可验证性 | 引用/标签/schema通过数；失败类型；NO_OP数 |
| 机制多样性 | 操作改变是否相同；去重的模型判断和不确定性单独记录 |
| 可操作性 | 是否明确 WHEN / WRONG / RIGHT / UNLESS / EXPECT |
| grounding | 每个结论的消息位置/hash；观察事实、推测、替代解释分开 |
| 重复修复 | 是否重复原有拒绝机制；是否有实质新证据或新操作改变 |
| 成本 | 各阶段 calls/tokens/latency，整个 replay wall-clock |

如果得到零个合规候选，报告“本次同证据生成未提出合规修复”，不补凑第三个。
如果有合法候选，也不能仅凭生成对照认定有效；下一步仍由真实 Gate 决定。

默认准备命令（不收费）：

```sh
cd /Users/spring/RSI/Evotau-analyst
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python \
  experiments/execution/replay-airline-skill-generation.py \
  --source experiments/replay-inputs/airline-gen0-e10-service-context.json \
  --output experiments/diagnostics/airline-analyst-prompt-comparison
```

未来明确授权后，官方 DeepSeek key 临时注入 OPENAI_API_KEY，同一命令增加 `--execute`。
不要把 key 写入协议、命令日志或版本库。当前不执行。
replay 目前只支持生成对照，不包含 Skill Validator 或 Screen。不能把它称作端到端效果实验。

## 步骤 B：独立新版 G2 进化运行

步骤 A 用于观察与决定是否启动，不用于修改本轮已冻结的 Prompt/阈值后还声称同一协议。
推荐运行完整的新版配置，保留正常 Customer–Service alternating，而不是关闭 Customer 偷换研究问题。

每代流程：
1. 当前 C/S 的 E10；合法 Customer candidate 的 E10；按原有 accuracy/tie 规则选择 Customer。
2. 只给 Analyst 当前 E 的可见证据和 E-only 历史；严格检查 task×seed 标签、引用和 hash。
3. 分配去重后的固定假设；每个 Mutator 可以生成候选或 NO_OP，不能换根因。
4. 合规 Skill → 原 Screen → Full-E 效果记录 → 原 V-primary 多对手 Gate。
5. 按原有选择规则提交一代结果；记录所有未部署 archive 候选与部署 Skill 的区别。
6. Gen1 使用上一代实际提交的 C/S，所有数据边界和恢复规则不变。

E 只用于生成、初筛及效果分析。V 仍是部署决定的主要依据：当前 Customer 要求 superiority；
历史/native Customer 要求 preservation；合法回归容许和严重违规/stuck约束完全按现有代码。
示例阈值：95% confidence，min_tasks=20，bootstrap=2000，Bonferroni，min_success_gain=0.02，
max_harmfulness=0.15，max_stuck_delta=0.05，max_stuck_rate=0.2，严重违规必须为0。
不把“E多成功两次”或“V多成功一次”直接等同晋升。

同条件 native episode 可以考虑复用，但必须先有 compatibility audit 和导入证据；
本协议准备阶段没有自动导入分数。Customer/Service状态、模型args、seed、原生运行设置或评测改变，
就不能复用。旧 Evolver 输出不能当成新 Analyst 缓存。

当前 replay 脚本尚未提供将其生成候选作为完整 G2 固定 proposal 导入的接口。
因此按现有正式 launcher，步骤 B 会重新生成候选；步骤 A 的生成不能宣称就是 B 的候选。
若未来希望直接评测 replay 候选，必须先实现并离线验证独立的冻结候选导入 runner，另记 provenance；
不能手改 journal 假装某个正式请求已经完成。本轮优先沿用现成完整 launcher。

## 步骤 C：最终 H scorecard

冻结 ST 和一次 E-only fresh Customer proposal 后才加载 H。
报告 S0/ST 在 native Customer 与合法 fresh Customer 下的 H20×4；同条件相同 S0=ST 则明确复用。
若 fresh Customer 不合法，原样记录 unavailable，只报告 native H，不冒充 fresh、不反复生成。

H 只能用于最终报告，不传回 Analyst/Mutator、Screen、Gate或下一代搜索。
当前旧版实验也使用同一 H20：本协议、Prompt和面板必须在查看旧版 H 结果前冻结。
若已查看旧 H 并据此调过新版，则共享 H 只能称 reused scorecard，不能称新的独立测试集。
V 被多候选多代反复使用，不能称完全独立泛化验证。H20同样不足以证明很小的总体风险上限。

主要终点：H成功率、Pass¹、Pass⁴、stuck rate、原生任务违规及成本。
辅助终点：晋升数、每代E/V配对净收益、修复/回归、seed稳定性、激活率、未激活条件的反转。
最终同时提供 task-block 区间和逐任务结果；不把四个seed当80道独立题。

## 执行量与时间解释

| 项目 | 场数 |
|---|---:|
| 一代原始 E fitness | incumbent10；每个合法 Customer candidate再10 |
| 一个 Skill 的 Full-E 对比 | 10×4×2=80（相同旧版可缓存复用） |
| 一个 Skill、一个对手的 V 对比 | 20×4×2=160（相同旧版可缓存复用） |
| 最终 native H，S0不同于ST | 20×4×2=160 |
| 最终 fresh H可用且S0不同于ST | 再160 |

初筛任务数是动态但有既定规则；其他对手、crossover和候选是否进入V也是条件性的。
原版V可按严格相同条件摊销，但每个新Skill的候选V一般需新跑80场。
因此新版也可能需要数百场对话；分析阶段减少几个候选并不保证总费用或wall-clock大幅下降。
以当前观察到约50–60场/小时仅作调度参考，不作完成时限承诺。原生episode和Activator是成本大头。
不为赶时间在中途减少seed、改Gate、提高并发或修改冻结配置。

## 失败、恢复与最终报告

- 普通 native benchmark 失败仍计入原始accuracy，继续运行。
- 候选/Analyst/去重合法JSON但违反契约：记录拒绝；不改写输出、不盲目回退、不强制产生Skill。
- 网络、JSON解析、runtime错误：显式失败并保存阶段、错误类型、输入输出及可能已收费请求；不记作任务0分。
- 不自动重试、切模型或换provider。原配置显式resume复用完成的calls/stages/episodes。
- 修改模型或协议后创建新配置、manifest和目录，不覆盖旧实验。

每代交付 C前后accuracy、候选机制/证据、E修复与回归、各对手V verdict、S是否部署更新。
最终交付 native/fresh H 的 S0/ST、四seed稳定性、激活与未激活反转、calls/tokens/阶段wall-clock、
完整拒绝和异常记录，以及commit/config/manifest身份。零晋升也完整交付。

## 当前可执行状态与授权边界

已有：331离线测试通过；独立版本/配置；真实Gen0冻结输入；不收费的replay准备已完成。
尚无：新版在线生成质量和真实修复效果；A/A阈值校准；replay候选导入完整G2的接口。
本协议是待审阅草案，不构成新版收费运行授权；当前旧实验继续。

正式启动命令形状（本次不执行；需用户授权后再记录实际launch和manifest）：

```sh
cd /Users/spring/RSI/Evotau-analyst
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python \
  experiments/execution/run-airline-gateway.py --mode formal \
  --config configs/airline-failure-analyst-qwen37plus-official-dsflash-p2.yaml \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074 \
  --execute --approve-unbounded-requests
```

推荐顺序：旧实验继续 → 同证据生成对照 → 看证据/多样性/合法性 → 授权独立新版G2 → 冻结后H。
没有把同一批V/H的已知结果拿来挑新版Prompt；没有新增收费预演阶段。
