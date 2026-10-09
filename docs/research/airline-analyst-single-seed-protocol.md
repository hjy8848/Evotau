# EvoTau Airline：单 seed 新版实验协议

状态：2026-10-09 用户已选择单 seed；配置和协议已准备，尚未启动收费运行。
本协议替代新版四 seed 草案的运行安排，不修改正在运行的旧版四 seed 实验。
代码生成算法仍是 `analyst_skill_v_validation_v3`（007fed9实现）；只改变新版重复评测次数。

## 目标

验证 Failure Analyst → distinct hypotheses → assigned Mutator 是否能提出不同、有证据、可执行的修复，
以及这些候选能否通过现有 Screen / V Gate，积累 SkillMemory。
这是单 seed 的探索性机制实验，不是多 seed 稳定性或严格总体泛化证明。
不会为了产生晋升而放宽阈值；零假设、NO_OP、拒绝、INCONCLUSIVE和零晋升都是有效结果。

## 冻结配置

配置文件：`configs/airline-failure-analyst-single-seed-p2.yaml`。

| 项目 | 设置 |
|---|---|
| 领域 | 原生 τ-bench Airline，upstream b7ea9074 |
| 面板 | E10 / V20 / H20，沿用现有固定IDs，彼此无重叠 |
| generation | 2（Gen0、Gen1） |
| Customer candidates | 每代1个 |
| Skill proposals | 每代最多3个不同假设，各1个候选；保留最多1个条件性 crossover |
| 所有随机运行条件 | seed=1；evolution_fitness_seed=1；screen_seeds=[1]；gate_seeds=[1] |
| 并发 / steps | episode并发2，单场turn串行；max_steps=200 |
| runtime模型 | Agent/Customer/Evaluator/Activator：gateway openai/dashscope/qwen3.7-plus |
| runtime args | temperature=0、enable_thinking=false，与新版四seed配置相同 |
| Evolver | 官方 openai/deepseek-flash，thinking enabled，reasoning_effort=high，原args不变 |
| 预算 | request_budget_cap=null，不额外设置请求、输出token或上下文上限 |

E：`1,0,39,49,11,27,34,12,7,15`。
V：`41,20,17,40,23,33,4,10,47,28,36,46,3,14,5,21,43,9,42,38`。
H：`2,6,8,13,16,18,19,22,24,25,26,29,30,31,32,35,37,44,45,48`。

独立experiment ID：
`evotau-airline-failure-analyst-qwen37plus-official-dsflash-e10-v20-h20-g2-p2-single-seed-20261009`。
output/checkpoint路径使用该ID，不能续接或覆盖四seed实验manifest。
原生policy、tools、任务、evaluator、Customer约束、Activator和上下文压缩全部保持不变。

## 实际流程

Gen0：
1. C0/S0运行E10；合法Customer候选再跑E10；按原accuracy最小化与tie保留规则选择C1。
2. Analyst仅阅读选中Customer下的E单seed证据：完整结果概览＋原有代表性证据＋E-only历史。
3. 引用/schema校验与机制去重，得到0–3个固定假设；每个Mutator生成一个Skill或NO_OP。
4. 每个合规候选先过原有语义校验，然后单seed Screen；通过才跑Full-E和V。
5. 原V-primary规则决定晋升，提交C1/S1；未晋升候选仍保存Archive。

Gen1：C1/S1重复同样流程，得到最终CT/ST。每代不能因没有Skill晋升而补生成额外候选。

最后：冻结CT/ST和一次E-only fresh Customer proposal之后，才加载H20。
分别评测S0/ST的native H；fresh Customer合法且不同于既有Customer时再评测fresh H。
fresh被拒绝就记录unavailable，不反复生成、不把旧Customer冒充fresh。

## 晋升规则和单 seed 的统计解释

E用于生成、初筛和效果归因；V决定部署，H只做最终scorecard。
不向Analyst/Mutator提供V/H轨迹、隐藏scenario、参考动作或gold答案。
当前Customer的V比较仍要求superiority，历史/native仍要求preservation。
严重违规、stuck、回归约束、bootstrap、Bonferroni和原阈值均保持不变。

只把seed数组由原[1,2]/[1,2,3,4]改为[1]，不降低min_tasks=20、95%置信度、
min_success_gain=0.02、max_harmfulness=0.15等门槛。配对仍按task×seed；bootstrap仍以task为块。

注意：20个V任务在一个seed下仍是20个task块，但无法估计同task的跨seed波动。
min_positive_seed_fraction退化成唯一seed是否有正收益，不再代表跨seed稳定性。
单seed Screen的比例/计数保护仍按原规则执行，分母变小后有效容许量可能变化；
不能声称重复次数减少后Gate的统计表现与四seed完全等价。保留实际判定和中间统计。
A/A calibration仍未完成，calibration_confirmed=false；未把选择单seed假装成噪声校准。
模型即使temperature=0，同task×seed的独立对话仍可能不同；修复/回归是关联证据，不是单Skill因果证明。

## 任务对话数量

| 阶段 | 四seed新版草案 | 本单seed协议 |
|---|---:|---:|
| E生成fitness，单个C/S条件 | 10 | 10（原本就是单seed） |
| Screen，选中n个task，原版+候选 | 4n（两个seed） | 2n |
| 一个Skill Full-E，原版+候选 | 80 | 20 |
| 一个Skill/一个对手 V，原版+候选 | 160 | 40 |
| native H，S0与ST不同 | 160 | 40 |
| fresh H可用且S0与ST不同 | 再160 | 再40 |

相同原版条件可复用，多个对手可能是同一条件；不按stage标签重复执行。
不同候选的Service条件不同，不能因为seed相同就复用其结果。
单seed最终H实际独立场数通常在20–80：S0=ST且无fresh为20；两个不同Service×两种Customer为80。
Screen不过关时不跑Full-E/V，其他候选是否进入V是条件性的，因此总场数不是固定数字。
主要重复评测阶段减少75%，Screen减少50%；生成没有减少。整场wall-clock不会精确快4倍。
不能用单seed成绩和旧四seed均值直接宣称模型能力提升，也不能按总成功次数跨条件比较。

## 结果表

每代报告：Customer前后Eaccuracy、实际假设/候选数、拒绝/NO_OP、Skill目标与风险，
Screen/Full-E/V的修复、回归、配对净收益、各对手verdict、是否部署、Skill数。
最终报告：S0/ST native H和fresh H的单次成功率（Pass¹）、stuck、终止原因、严重违规、
Skill激活率和未激活条件反转、API calls/tokens、各阶段wall-clock、总实际episode数。

**Pass⁴与跨seed稳定性必须显示“未测量”，不能由单seed复制、推算或冒充。**
任务块区间可以报告，但限定为这一面板/执行条件；V被反复使用，存在选择偏差。
新旧均共享H20：新版Prompt/协议应在查看旧H之前冻结；若已据旧H调参，
这批H只能标为reused scorecard，不能作为新的独立测试证明。不能根据H回选Skill。

## 恢复和成本

普通benchmark失败正常计分；已解析但不合规候选记录拒绝继续，禁止静默改写输出或强制Skill。
HTTP/JSON解析/runtime异常明确失败，不伪造成task0分；保存原输入输出、阶段和收费请求证据。
不自动重试或换模型；显式同冻结配置resume复用完成的episode、stage和请求。
没有自动导入旧baseline分数；如需复用必须先做完整兼容审计并保留导入证据。

可选的同证据生成replay仍是0场episode、4–8次Evolver请求；不是启动前强制新增预演。
已有replay不能直接注入本G2：现成正式launcher会重新生成候选，不能手工篡改journal。
运行后是否值得做四seed确认，依据本轮结果讨论并另建协议，不在本轮临时改seed。

## 命令与当前状态

只做配置dry validation（不收费）：

```sh
cd /Users/spring/RSI/Evotau-analyst
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python \
 experiments/execution/run-airline-gateway.py --mode formal \
 --config configs/airline-failure-analyst-single-seed-p2.yaml \
 --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074
```

未来用户明确要求开跑后，在同一命令加 `--execute --approve-unbounded-requests`，
由原有launcher读取已有凭证并记录实际manifest/commit/PID。本次没有启动。
当前旧版四seed实验保持运行；准备单seed协议不等于修改它或把它标为新版。
