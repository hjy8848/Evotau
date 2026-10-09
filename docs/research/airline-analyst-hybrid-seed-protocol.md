# EvoTau Airline 新版混合 seed 实验协议

2026-10-09。用户已同意 E/初筛单 seed、V 两 seed、H 单 seed 的协议方向。
本轮完成代码、独立配置和离线验证；尚未发新版收费请求。当前四 seed 实验继续原样运行。
本协议替代不可启动的全单 seed 草案；旧草案保留为已撤回的审计记录。

## 目标与解释范围

在相同原生模型与 Airline 任务下，验证 Failure Analyst → 机制去重 → assigned Mutator
能否提出有证据支持、不同操作机制的修复，并经原有 V Gate 晋升与积累 SkillMemory。
候选拒绝、INCONCLUSIVE、NO_OP和零晋升均合法。
新版同时改变生成分工、机制去重和 Validator 的领域复用表述，不能把结果全归因于 Analyst 单一因素。
这是探索性机制实验；A/A尚未校准，H只测一个seed，不能报告多seed稳定性或严格总体泛化证明。

## 冻结安排

| 阶段 | 面板 | seeds | 用途 |
|---|---|---|---|
| 生成 fitness | E10 | [1] | Customer选择、失败证据 |
| Screen | 既定E目标与成功控制 | [1] | 候选初筛 |
| Full-E | E10 | [1] | 记录修复/回归及收益 |
| V Gate | V20 | [1,2] | Skill晋升，按原task-block统计规则 |
| Final H | H20 | [1] | 冻结后的native/fresh scorecard |

E10：`1,0,39,49,11,27,34,12,7,15`。
V20：`41,20,17,40,23,33,4,10,47,28,36,46,3,14,5,21,43,9,42,38`。
H20：`2,6,8,13,16,18,19,22,24,25,26,29,30,31,32,35,37,44,45,48`。

其他冻结项：G2；每代Customer候选1；最多3个不同假设各生成1个Skill；保留最多1个条件性crossover。
并发2，单场turn串行，max_steps=200。原生τ-bench Airline upstream b7ea9074、policy/tools/backend/
evaluator均不改。runtime Agent/Customer/Evaluator/Activator继续gateway
`openai/dashscope/qwen3.7-plus`，temperature=0、enable_thinking=false。
Evolver继续官方 `openai/deepseek-flash`，thinking enabled、reasoning_effort=high，原model args不变。
request_budget_cap=null；不自行新增请求、输出token或上下文上限。

配置：`configs/airline-failure-analyst-hybrid-seeds-p2.yaml`。
独立ID：`evotau-airline-failure-analyst-qwen37plus-official-dsflash-e10-v20-h20-g2-p2-hybrid-seeds-20261009`。
output/checkpoint用此ID；不覆盖、续接或冒充当前四seed实验。

## 完整流程

每代：C/S运行E → 合法Customer候选E → 原accuracy/tie选择规则 → Analyst只读选中E证据与E-only历史
→ 严格引用/标签检查 → 机制去重与固定假设分配 → Mutator候选或NO_OP → 原Validator
→ 单seed Screen → 单seed Full-E → 双seed V多对手Gate → Archive与generation commit。
Gen1使用Gen0实际提交的C/S，不能为填满候选预算制造假设。

Analyst不生成Skill；Mutator不能换根因/目标，证据不足可NO_OP。
Service侧不得读取hidden Customer scenario、gold或V/H反馈；不能从Archive自由文本间接泄漏。
原语义/权限/政策/证据校验、Activator、Screen阈值、V-primary和多对手规则保持不变。
当前Customer的V比较要求superiority；历史/native要求preservation。
Gate仍要求至少两个配对seed；V两seed满足必要覆盖条件，但不保证通过。
min_tasks=20、95%confidence、bootstrap=2000、Bonferroni、min_success_gain=0.02、
max_harmfulness=0.15、max_stuck_delta=0.05、max_stuck_rate=0.2、零严重违规等阈值都未降低。
Screen改成单seed会改变比例/计数规则的实际分母，记录实际保护条件，不宣称统计性质完全等价。
同task的两个seed不能当两个独立task；bootstrap仍按task块。

## H隔离与报告

冻结ST并完成一次E-only fresh Customer生成及合法性检查后才加载H。
fresh不合法/不distinct就记unavailable，只报告native H，不重复生成或拿旧Customer冒充。
S0=ST、Customer及所有运行条件也相同则明确复用；不能为不同Service复用旧分数。
H单seed报告Pass¹、成功率、stuck、违规、token与配对反转；Pass²/Pass⁴和跨seed稳定性为未测量。
V被多代多候选复用，有选择偏差；H20单seed是scorecard，不是强风险上界证明。
旧实验与新版共享H20，Prompt/协议应在查看旧H并据其调参前冻结；若做过这类调参，
则标为reused scorecard而非新的独立测试。禁止H回流选择/调参。
配对评测仍是独立对话，成功反转不自动证明Skill因果作用；单独报告未激活条件反转。

## 实际工作量

| 每个比较 | 本协议 | 原四seed新版 |
|---|---:|---:|
| 一个Skill Full-E，原版+候选 | 20场 | 80场 |
| 一个Skill/一个对手 V，原版+候选 | 80场 | 160场 |
| native H，S0与ST不同 | 40场 | 160场 |
| 合法fresh H，S0与ST不同 | 再40场 | 再160场 |

Screen选n个task，两个Service共2n场，原来两个seed为4n场。
E生成fitness本来单seed，不减少；Analyst额外1次，有多假设时去重额外1次，Mutator0–3次。
旧版同条件可缓存摊销；每个不同候选通常仍需新的V40场。
多个对手/crossover/初筛拒绝使总量可变，不能预先宣称固定总episode数或严格快四倍。
主要V工作量减半，Full-E/H减为四分之一；整体时间取决于候选筛选和provider延迟。

## 工程实现与兼容性

新增可选字段 `evaluation.repair_seeds` 和 `evaluation.heldout_seeds`：分别路由Full-E与H。
`gate_seeds`只控制V（旧接口H函数参数仍叫gate_seeds，但接收的是解析后的H安排）。
`screen_seeds`与fitness seed保持独立。新字段显式进入新manifest/hash。
旧配置不含新字段时，不自动补写字段，运行回退到旧gate_seeds，不重解释历史manifest。
全单seedV-primary配置仍在任何provider请求前拒绝启动。
原本保存的native episode只能在明确兼容审计后导入；本次没有自动导入任何分数。
所有成功stage、Evolver请求和episode在同冻结配置下精确resume，不重复已完成条件。

普通benchmark失败计入原指标并继续；合法JSON但契约无效记录拒绝；网络/JSON解析/runtime异常
明确失败保存证据，不伪造task0分、不静默修补、不自动重试/换模型。
需要模型或协议改变时另建manifest/目录。没有晋升也完成两代和可用H报告。
最终逐代给出C选择、候选机制、Screen/E/V收益与拒绝原因、部署变化，最后给S0/ST H、
激活与未激活反转、API调用/token/各阶段wall-clock及完整provenance。

## 命令（先准备，不收费）

```sh
cd /Users/spring/RSI/Evotau-analyst
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python \
 experiments/execution/run-airline-gateway.py --mode formal \
 --config configs/airline-failure-analyst-hybrid-seeds-p2.yaml \
 --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074
```

用户明确要求启动后，同一命令增加 `--execute --approve-unbounded-requests`；原launcher读取既有凭证。
可选的Gen0同证据生成replay仍是0个episode、4–8次Evolver请求，不强制新增收费预演。
既有replay不能手改journal导入完整G2，本配置正式启动时会正常生成自己的候选。
本次未启动；当前旧实验不切换、不停止。

离线验证：完整pytest 340项通过（6项依赖弃用警告），src/tests Ruff及git diff --check通过。
新版正式模式dry validation通过，real_requests_started=false。包含E/V独立路由、H单seed及Pass4未测量、
非法seed数组拒绝、旧配置回退/序列化兼容、单seed晋升拒绝等回归测试。
