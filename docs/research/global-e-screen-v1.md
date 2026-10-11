# Global E Screen v1

普通 Service 进化新增可选筛选协议；旧协议和已冻结 Discovery 定向修复保留。

```yaml
skill_evolution_v2:
  evaluation:
    promotion_protocol: v_primary
    screen_protocol: global_e_v1
```

省略 `screen_protocol` 仍使用 `target_fix_v1`，规范化配置不插入该字段，历史配置序列化不变。显式字段只能是 `target_fix_v1` 或 `global_e_v1`，后者仅允许 V-primary。新字段进入 Manifest / checkpoint 身份和显式 Screen stage 输入；两种协议不能混合 resume 或共用已完成候选评测。

## 精确规则

`cheap_screen()` 不变。`global_e_screen()` 复用其 task×seed 配对校验、重复 cell 拒绝、未知结果拒绝、成功 cell 保护、硬违规增量和 max_steps 风险统计。

- 净改善 = 全部 Fail→Pass cells − 全部 Pass→Fail cells，必须严格大于 0。
- 回归数不超过 `max(screen_regression_allowance, screen_max_regression_rate × 原成功 cells 数)`。
- 硬违规增量必须为零。
- stuck 数增量 / 配对 cells 数不超过 `screen_max_stuck_delta`。
- 任一未知结果不得算失败或收益；空面板不能通过。
- `fixed_cells` 仍表示指定目标修复数，仅作诊断；新增 `overall_fixed_cells`、原始 cell 引用、`net_gain`。

与旧协议相比，唯一接受条件变化是用整体净改善替代 `fixed_cells > 0`。没有降低其他风险限制。这里的“整体”指当前既有 E Screen 子面板，不声称低成本子面板代表完整 E；通过后仍执行完整 E 配对记录和独立 V Gate。没有增加候选排序、Top-K 或新的晋升路径。

存在 `frozen_targets` 时，无论全局配置如何，实际 Screen 协议强制 `target_fix_v1`，继续执行原 `evaluate_target()`、E 成功保护和 V 检查。其他任务改善不能冒充 Discovery 被修复。

V `evaluate_gate()`、多对手 replay、superiority/preservation、bootstrap、H 延迟加载、Customer Prompt、Activator、原生 τ-bench policy/tools/evaluator 全部不变。V 拒绝或证据不足时不部署 Skill。E 改善不等于因果效果、目标修复或晋升。

## 新配置

`configs/airline-global-e-v1-hybrid-seeds-p2.yaml` 从历史 Analyst recovery 配置复制，只新增筛选身份、独立 experiment/output/checkpoint 和准备状态；没有导入旧分数。

- E10 / V20 / H20 原任务划分不变，G2 / P2 / max_steps=200。
- E、Screen、H seed 1；V seeds 1、2。
- Qwen3.7-plus 原生 runtime / Activator，官方 DeepSeek Flash reasoning high Evolver。
- V 原阈值不变：bootstrap 2000、confidence .95、min_tasks 20、min_gain .02、max_harmfulness .15、max_stuck_delta .05、max_stuck_rate .2、零硬违规，多次查看 Bonferroni。
- 原 Customer prompt 和 representation 不变；不借这次修改启用 Prompt V2。
- 保留用户既有无上限预算偏好，但本任务未授权发起该新收费实验。

只做启动检查：

```bash
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python \
  experiments/execution/run-airline-gateway.py --mode formal \
  --config configs/airline-global-e-v1-hybrid-seeds-p2.yaml \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074
```

收到独立启动授权后，才增加 `--execute --approve-unbounded-requests`。

## 验证与限制

新增测试覆盖历史 Task12/34 分流、整体退化/平局、有限回归、未知结果、硬违规、stuck、重复 seed、task×seed 保护、旧配置兼容、Manifest 隔离、G2 / resume、不部署 V 拒绝候选，以及 Mock V 通过才能晋升。Mock 小面板参数仅用于控制流测试，不改变正式配置。

现有 Targeted Repair / Retail / Airline / E/V/H 隔离测试全量运行，记录见 `research/audits/global-e-screen-v1/`。测试不是实证有效性证据。A/A 未校准、历史 H20 已被接触、H 单 seed 不报告 Pass4。观察到的 E 净收益仍可能来自在线随机波动；新协议实际进入 V 的频率、晋升率、调用成本尚未 live 验证。

实现于独立 worktree，原 A/B pilot 源码/HEAD/配置未动。
