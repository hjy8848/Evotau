# Global E Screen v1 离线验收

基线 f0fdabe55822cae137ca5595f61483191dbbc95f，独立 worktree `/Users/spring/RSI/Evotau-global-screen`。

- 新增 `global_e_v1` opt-in Screen；旧 `cheap_screen` 与 `evaluate_gate` 未修改。
- 原目标失败但其他 E cell 改善，可以进入 V；V 拒绝仍不部署。
- 冻结 Discovery targets 优先走原定向协议；全量测试包含定向修复保护。
- G2 mock 验证拒绝、归档、恢复无重复调用；另测 V 通过才晋升。
- 新增 14 项测试，Ruff src/tests 和 diff check 通过。
- 首次全量测试 563 passed / 1 failed：独立 worktree 缺少 Git 忽略的历史 morphology JSON 工件。没有代码缺陷修补或测试跳过；复制相同原文件后该测试独立复跑通过，来源/hash 见 fixture-copy.json。最终全量复跑见 pytest-final.log；首次失败保留在 pytest.log。
- 四份历史配置的旧/新 freeze policy 相同，SHA 审计见 historical-policy-compatibility.json。
- 独立新配置 dry-run 通过，real_requests_started=false，见 dry-validation.log。
- 当前 A/B 目录的本地 HEAD 仍 f0fdabe，工作树干净，没有改变正在运行的源码、配置或调度。

没有发起真实 provider 请求，没有启动新实验或导入旧分数。真实筛选效率、晋升率及成本变化尚未验证。进入更多 V 评估可能增加费用；E 净改善仍可能受到在线噪声影响。原统计门槛和 A/A/H 限制不变。
