# 人工 Customer / Service Skill：授权范围与条件记忆

这是用户要求的人工候选诊断，不是自动进化或正式 Skill 晋升。

Customer：在原任务要求的信息问答打断操作讨论时，明确只批准当前选项，并返回尚未完成的原请求。保留全部原目标、披露时序、预算与支付事实，不教客服使用工具或绕过规则。

Service：维护每项操作的内部授权账本。写操作前核对预订、字段、完整费用、原条件和最近授权；信息插话不能转移授权或抹去条件。保持有效授权，避免无意义的重复确认。

独立 E3 诊断使用 Task 7、12、15，四组各三场：原版、只换 Customer、只换 Service、两边都换。seed=1，原生 max_steps=200，panel 内并发2。原模型、政策、工具和 evaluator 不变；V/H 不加载，无自动晋升，无旧分数导入。没有额外 API 探测。

先记录 Native Reward，再人工核查 Customer 忠实性、触发暴露、Service 激活、越权写操作和费用条件。单 seed 独立对话不能证明因果收益。作者已经看过 E 原生评价定义，因此本诊断不是公平盲测 Evolver 的模型能力。结构检查已通过；独立语义评审尚未完成，不能称已通过正式 Validator/Gate。

文件：`customer-skill.json`、`service-mutation.json`、`service-memory.json`；`evidence-rows.json` 保留原 E 消息索引和 SHA；`provenance.json` 标明作者与局限。`build.py` 离线重建并冻结 SHA。

```bash
PYTHONPATH=src LITELLM_LOCAL_MODEL_COST_MAP=True python experiments/execution/run-human-skill-factorial.py \
  --config configs/airline-human-scope-ledger-v1-e3-factorial-p2.yaml \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074 \
  --execute --approve-unbounded-requests
```

未完成请求保留响应和 episode 工件；启动器不会盲目重发失败运行。重新执行已完成运行只读取结果。
