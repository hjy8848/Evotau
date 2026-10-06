# 2026-10-06 审查问题修复

基于 `9353051872c76644dd9a2e96c2105acceac7ff96` 修复[历史审查](2026-10-06-idea-implementation-audit.md)中的 F1–F7。历史审查及真实实验不覆写。本轮仅工程修复、离线验证和文档更新，没有真实 provider 调用，没有启动 τ-bench experiment。

## 逐项修复与证据

| 审查项 | 修复 | 回归证据（`tests/test_audit_repairs.py`） |
|---|---|---|
| F1：runtime 未转换 thinking 参数 | Runner 与 Evolver 共用 `role_model_args_for_runtime`；Flash role 的 `thinking_mode: disabled` 转为 `extra_body.thinking.type: disabled` | 实际 Runner 参数 → 原生 τ `generate` → LiteLLM → OpenAI SDK 序列化，MockTransport 拦截三种 runtime role 的 HTTP 请求；均发送 `thinking.type=disabled`，没有 `thinking_mode` |
| F2：未知 native score 当作 failure | 无效 reward 保存为 incomplete/unscored，不创建完成 record、不缓存；panel 和 `_accuracy` 均要求有效 boolean score；缺失 trajectory/message list 不当作空对话 | None/bool/string/NaN/Infinity reward；UNCERTAIN fitness；missing trajectory；均停止评分/选择 |
| F3：commit/dirty 变化破坏 resume | 保留原始 immutable manifest 和 checkpoint identity；绑定保存的 Git metadata 时仍严格比较 source、selected-config SHA、model/args、pins、panels 等执行输入；每次 invocation 另记当前 Git provenance | CLI、原生 runner、Console preview 都能在仅 Git metadata 改变后续跑；改变 source/config/steps 或篡改 manifest 被拒绝；新增无关 config 不改变 source SHA |
| F4：失败证据与状态不足 | 每次 Chat/Responses 请求记录脱敏实际 args、响应 ID/model、finish reason/status、visible/reasoning/tool 状态、usage、latency 和异常；保存 partial trajectory、Evolver exact input/parsed output/failure；CLI 收尾为 failed/paused/complete | 空回复 API success 与语义 empty 分开；保留 `length`/`max_output_tokens` 和 reasoning token metadata；不记录 key/prompt/reasoning 文本到 call log；并发 log 隔离与 cap accounting；CLI failure/pause 和 KeyboardInterrupt artifacts；HTTP504 单次失败无 retry |
| F5：MultiToolMessage results 丢失 | 将嵌套 tool messages 展开为可见 tool results，继续剔除 raw provider data | 两个 tool results、tool_call_id/content 保留，private raw_data 不进入 Evolver context |
| F6：panel label 导致重复 rollout | cache key 为 task/seed/C/S，panel 仅作引用；每条件锁 coalesce 并发重复；cache lookup 在 reservation 之前；panel references 原子持久化 | 同一条件两个并发 panel 仅 dispatch 一次；cap 用尽后 cache reuse 无 reservation/denial；恢复后复用；三代 refs 指向一个 completed episode |
| F7：文档落后于实现 | README 与 method note 更新为四角色、native accuracy selection、aggregate V gate、E-only flags、SkillMemory、fixed seed、信息隔离、失败/缓存/续跑语义 | 文档与 active source 对照；Customer/Service Evolver prompt 内容和 selection rules 未改 |

## 同时修复的关联问题

- Console 原来只接受 schema v1，无法读取当前 CLI 写出的 schema v2 完成结果；现支持两种 alternating schema，Phase 0 的 schema 校验保持原约束。
- Console 展示失败前的 Customer/Service 气泡、工具结果和失败原因，结果保持 unavailable；复用 episode 支持按所有 panel/generation aliases 筛选，实际 episode 数量不重复。
- Console preview/display 使用 manifest 的真实并发数。
- E/V runner 恢复缓存时跳过非活动 panel 的 trajectory；已有 H conversation 只由完成 fresh challenge 后构造的 H runner 读取。新增 sentinel 回归证明 evolution resume 不读取 H 对话。
- JSON 更新使用独立临时文件、fsync 和原子替换。总 runtime 累计各 invocation 的已记录执行时长，另存包含两次启动之间等待的 elapsed time。

## 验证与边界

运行 `python -m pytest -q`、`ruff check src tests`、`git diff --check`。新增测试全部在本地拦截 provider 请求，既没有访问真实 InferAI，也没有消耗真实模型 tokens。完整测试 **114 passed**（新增 22 个回归 cases），ruff 与 diff whitespace checks 均通过。

Customer Evolver、Service Evolver 和 SkillMemory Evolver 的研究 prompt 不变；Customer strict-lower/tie-incumbent、Service strict E improvement、可选 V aggregate no-regression、phase/generation 顺序不变。没有新增 Judge/Reviewer、失败 taxonomy、validator 搜索机制、重试或模型切换。

这次修复改变了 runtime 实际参数及源码 identity。旧实验继续保留，不能把旧 cache 当作修复后条件的成绩。后续真实验证需要新的 output/manifest；此次没有创建或启动新实验。发送正确 `thinking` 参数仍不等于 provider 已遵守，必须检查后续真实 response metadata。

HTTP504 是 provider/网关层失败，代码无法保证消除；现会留下调用阶段和失败证据，并允许对不完整项显式 resume。已有空回复的确切原因仍未知。有效 Customer pressure、Service repair 和 H 泛化仍然是研究问题，不能从工程修复或离线通过推断出来。
