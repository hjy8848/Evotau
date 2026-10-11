# Customer Evolver Prompt V2: implementation and pre-registration

## 实现了什么

基线远程 `5c56f681225872431f8af4fb070332ae144be53f` 已核对。本地原工作树干净；目标分支在另一工作树中，修改保留在本工作树 `codex/airline-failure-analyst` 的本地提交，不覆盖其他工作树、不推送远程。本次没有真实模型请求。

新增可选字段 `customer_evolution.evolver_prompt_version: v1|v2`。缺省不添加字段，仍使用 V1 原始 Prompt 和调用身份，旧规范化配置不会悄悄变成 V2。普通、RC-Bandit、Open-RC 的 V1 Prompt 字节哈希有回归保护；Open-RC 改为显式的无示例核心，不再在运行时做字符串替换。V2 同时支持普通/Open-RC/RC-Bandit，各有 `_v2` 调用身份。Open-RC 路径不选 Arm、不调用 UCB、不添加模型角色。Customer Skill schema 仍为 v1。

新增独立可选字段 `trajectory_validator_version: v1|v2`。缺省保留 V1；v2 从保存的原始 native messages 构造编号视图，保留零基索引、源消息 hash 和可引用的用户索引，不展开 multi_tool 后猜测位置。错误引用继续拒绝/uncertain，不重映射历史模型输出。该修复独立提交，详见 `trajectory-index-diagnosis.md`。A/B 两组都使用 v2 校验，因此不是比较变量。

原 Customer INVARIANTS、Task/Reward、Service Screen/Gate、选择规则、E/V/H 边界、Repair Pair 与 Discovery/Targeted Repair 协议均未修改。当前仍没有正式晋升的 Repair Pair，本 Pilot 不是 Open-RC 方法效果实验。

## 为什么这样设计

V2 区分真实用户协作和主动提供 Service 内部操作方案：预算、必要报价和知情授权仍合法；不鼓励用户给工具序列、内部推理或解题清单。现有 mechanism/trigger/hypothesis/procedure 承载具体决策、失败预测、机制解释、合格客服行为和“测试而非辅导”的理由，不增加 schema 字段或固定方向。

这是生成目标修正，并非本地语义保证。合法 Customer 仍可能帮助 Service 成功，模型仍可能不遵守指令。实际选择必须依据完整原生 E 评测与合法性校验，不能为制造失败隐藏预算或改变授权。

## 证据采样审计（本次不改变）

全部 E10 结果可见，实际详细案例为成功 39/15 与失败 34。排序用工具错误、工具种类、WRITE 数量、工具数量及 task/seed 稳定次序。它保留成功/失败对照，但“复杂”不等于“机制相关”，并不能证明 Task12 应当被优先展示。当前数据不足以判定低相关案例占据了多少有效上下文；因此保持采样和 3 个案例不变，避免与 Prompt 单变量比较混杂。

独立后续 Proposal：离线标注原始可观察决策点覆盖与证据可还原性，比较复杂度采样和机制相关的确定性采样；不得按历史模型成绩挑题，不将其并入本次 A/B。

## 离线验证

`pytest-full.log` 保存完整实际 pytest 输出；`ruff-source-tests.log` 为源码和测试检查；四份 `*-dry.log` 是真实 dry-run（只加载 E）。`saved-context-offline.json` 使用同一保存 E Context 和同一归档模型输出走 V1/V2 provider 解析/结构契约，不是生成新候选。V1/V2 请求为 30,481/30,758 proxy tokens；均小于原有 40k 限制。没有添加上下文或输出上限。

测试覆盖：V1 三模式 exact hash、V2 六种显式 dispatch、非法版本/冲突模式、同 schema、Manifest/Checkpoint 不兼容拒绝、持久 Evolver cache 版本隔离、三类预算/依赖/纠错 fixture 的结构与原校验链、原始 multi_tool 索引、错误引用拒绝、G2 和 resume 不重复阶段。

Mock semantic verdict 只验证数据和控制流，不证明真实 Reviewer 语义判断正确，更不证明 Prompt 效果。全 `src/tests/experiments/execution` Ruff 发现 27 个既有历史脚本问题；涉及文件未被本次修改，见 `ruff.log`。未把这些无关历史脚本改造混入任务。

## 下一次真实 V1/V2 对照

四个独立配置：`configs/airline-customer-prompt-ab-{v1,v2}-e10-k2-seed{1,2}.yaml`。

- 固定 E：1,0,39,49,11,27,34,12,7,15（原 E10，未重新按表现挑选）。
- 每个配置 G1/K2，seed 1 或 2，P2，max_steps=200。
- 原生 Agent/Customer/Evaluator：`openai/dashscope/qwen3.7-plus`，原网关与参数不变。
- Evolver：`openai/deepseek-flash`，官方 DeepSeek thinking enabled / reasoning_effort high，原参数不变。
- 原生空 Service 冻结；V/H 不加载，不生成 Service Skill、不晋升、不运行 Repair feedback。
- 两组仅改变 Evolver Prompt 版本；每组两次独立生成和 rollout。不能保证同 task/seed 产生相同实际对话；不能据此声称因果收益。
- 预注册 `pilot-preregistration.json`：有效挑战率、新增失败、恢复条件、独立盲审机制具体性/可检验性与程序性协助率、成本。合法询价/披露预算/知情授权不单独算“协助”。未盲审时 assistance rate=N/A；unknown/invalid 不能计入有效发现。输出均保持原生 reward。

### Dry-run（可执行，无收费）

```bash
cd /Users/spring/RSI/Evotau-analyst
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python \
  experiments/execution/run-customer-skill-diagnostic.py \
  --config configs/airline-customer-prompt-ab-v2-e10-k2-seed1.yaml \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074
```

### 收费 Pilot 命令（仅准备，必须另获批准后才执行）

四个进程依次执行，避免叠加并发；每个进程内部 P2。保持同一冻结提交、配置与 output；故障时依原 request journal/checkpoint 审核恢复，不盲重发未决请求。

```bash
cd /Users/spring/RSI/Evotau-analyst
for seed in 1 2; do
  for version in v1 v2; do
    PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python \
      experiments/execution/run-customer-skill-diagnostic.py \
      --config "configs/airline-customer-prompt-ab-${version}-e10-k2-seed${seed}.yaml" \
      --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074 \
      --execute --approve-unbounded-requests || exit
  done
done
```

本任务未授权执行这段收费命令，亦未激活监控自动启动。保持 request_budget_cap=null，不擅自添加新上限。最大计划为 120 episodes（40 baseline + 80 candidate）；4 个生成请求、最多 8 静态与 80 轨迹校验请求。不合规候选可能减少 rollout；网络失败可能留下已收费的未决调用，因此这不是金额硬上界。

原 E10 空 Service 20 episodes 的样本为每 episode 9–21 次 runtime 请求、均值15.1。120 episodes 的样本外推为 runtime 1,080–2,520 次，均值1,812；另有最多92生成/校验请求。原生 runtime 均值外推约10.0M输入/0.36M输出 token，Evolver 与 Reviewer token 额外计账。max_steps=200 时可能更高，不把样本最大值称为严格上限。网关人民币费率未知，费用及人民币单位发现成本=N/A；见 `cost-projection.json`。零有效发现时单位发现成本报告 N/A，而不是0。

## 仍未验证

V2 是否增加合法新失败、减少不必要指导、Reviewer 是否更可靠引用索引、同模型多次运行波动幅度，均未真实验证。两次生成复制只是探索性 pilot，8个候选不能证明分布优势或总体泛化。H 已关闭；没有 Repair Pair 就不能声称 Repair-Conditioned Discovery。历史 uncertain 和 no-promotion 均保持原样。
