# First formal Retail SkillMemory run — stopped on InferAI group RPM limit

**Status: failed / 未完成，不能作为 accuracy 或正式 mechanism evidence。**

| 条件/阶段 | 完整 episodes | Accuracy | 结果 |
|---|---:|---|---|
| Gen0 C0 × S0 | 0/20 | N/A（未知，不是0%） | InferAI 429中止 |
| Gen0 Customer candidate/selection | 未启动 | N/A | 没有proposal或selection |
| Gen0 Service ADD/UPDATE/NO_OP | 未启动 | N/A | 没有operation或candidate replay |
| Gen1 全部阶段 | 未启动 | N/A | 没有cache reuse或新panel |

## Run validity

- experiment ID: `evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g2-p4-formal-20261006`
- run source commit: `03cae4aa28a677540a7c2f516edfaa38a8cb42be`；release canary源码来自`cda8ca8`，其后`03cae4a`只有归档变更。
- source SHA256: `965497fcfd0260fdf20cc2525fecc024d8853bd82d1baa44730f1b0ce3902f3e`
- manifest SHA256: `464b25a5b6ef240c85b8b70d817be08d8e03c94162e0bfef45aa62e0ba4afc7b`
- config SHA256 (canonical parsed JSON): `4a9505cc007f89cdac065a500d7109eeebfcff0e3e0302e01ebfdcf4492ea77d`
- frozen P=4、E20、G=2、candidate=1、seed/fitness seed=1、max_steps=32、预算无上限；没有源码/prompts/selection改动。
- E20固定为 `66 92 29 67 106 22 69 98 93 88 4 21 8 54 107 48 52 80 35 16`，20个unique合法train IDs；V/H/excluded互斥。实际只加载E20；V/H内容均未加载。
- preflight: 114 tests passed；`ruff check src tests`和`git diff --check`通过。
- status=failed；resume_count=0。初始SkillMemory=[]；final active memory未产生。skill count trajectory只有初始`0`，不能写成两个已完成generation的`0→0→0`。

## 真实失败与fail-closed

首批4个episode发生真实请求。在30个成功回复后出现分组RPM限制：

```text
HTTP 429
error.type = rate_limit_exceeded
message = group requests-per-minute limit exceeded
```

这条错误不能证明provider实际配置的精确RPM上限或reset时刻。所有5个失败attempt均保存partial trajectory、provider调用日志/response metadata、异常链和本episode usage。失败task为`66,92,29,67,106`；task106在worker释放、主线程取消排队任务之间已开始，故有5个累计attempt，**没有5路并发**；余15个排队job取消。所有分数均缺失且未进入fitness，未计作task failure；没有生成STOP、Customer fallback、proposal或selection。

没有completed episode、panel references或checkpoint，`alternating-result.json`也不存在。原生checkpoint只有完整generation才提交，未伪造completed_generation。原始失败run目录保留，未来显式resume使用同一frozen manifest/config，只复用真正完整condition；本次没有完整condition可复用。

## 参数与并发证据

35/35个实际Flash HTTP请求`thinking.type=disabled`；30个HTTP200回复reasoning_tokens全0，reasoning文本全无，异常空completion=0。5个HTTP429错误体没有completion，**不能归入“空模型回复”**，reasoning usage也不能替这些错误体假定为0。Pro calls=0，未进入Evolver。`reasoning_effort=high`只是固定请求配置，仍无provider独立high effort验证。

Wire时间序列观察到**4个不同episode worker同时处于HTTP请求中**；每个worker内部请求严格串行。executor max_workers=4，未调整并发。运行在首panel中止，native `EpisodeJobTelemetry`没有最终落盘；summary将native peak字段保留null，并单独保存实测HTTP peak=4证据，不伪造runtime telemetry。

35条wire request/response与35条episode provider日志、预算attempts、per-episode计数一致；所有30个成功response IDs唯一且只归属一个episode；失败和usage分别对账；in_flight=reserved=0；num_retries全部0，没有隐藏transport重试。独立backend状态由冻结runner为每episode独立构建environment并deepcopy task保证，场景source/runtime hashes一致；此次没有完整episode，因此不能实测成功condition缓存复用、selection order或跨generation顺序结果。

## API与耗时

| Role | Calls | Success | Fail | Prompt tokens | Completion tokens | Sum seconds | Mean seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| customer | 14 | 12 | 2 | 6,303 | 505 | 56.20 | 4.01 |
| service | 21 | 18 | 3 | 90,771 | 1,344 | 75.55 | 3.60 |
| evaluator | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 |
| customer_evolver | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 |
| service_evolver | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 |
| reviewer | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 |
| customer_judge | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 |
| service_judge | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 |

合计**35 calls（Flash35、Pro0），30成功、5失败；97,074 prompt tokens、1,849 completion tokens，共98,923 tokens；wall clock 38.82秒；sum API time 131.75秒**。失败请求usage不可用，token合计只包含已报告usage的30个成功请求。完整episodes=0，incomplete attempts=5，panel references=0，cache hits=0，new started episode jobs=5，scheduled20/cancelled15。Timeout/5xx=0，429=5。

与canary并发1的13分15秒不能计算speedup：本轮只运行了38.82秒即失败，工作量和完成度完全不同。没有正式运行时长或有效accuracy曲线。

## Evolution解释

Customer challenge、reusable skill、ADD/UPDATE/NO_OP credit assignment、repaired/regressed task IDs、paired pass/fail transitions和`challenge→repair→new challenge`均**未产生，无法判断**。当前直接瓶颈是provider请求速率限制，不能据此归因Customer/Evolver能力、Skill representation、Flash能力或统计噪声。V/H关闭、seed1、Retail only、无matched baseline的研究边界保持不变。

## 文件与后续恢复

- `formal-failure-summary.json`：可机读统计、证据检查和不可推断项。
- `actual-provider-http.jsonl`：脱敏实际request/response（不含key/headers/messages/reasoning文本）。
- `episodes/*/partial-simulation.json`、`incomplete-run.json`、`provider-calls.jsonl`：5个真实失败attempt。
- `manifest.json`、`run-context.json`、`run-execution-state.json`、`config.yaml`、`preflight.json`：冻结输入、启停状态与启动前校验。
- `execution-observer.py`：此次实际调用入口；只观测HTTP元数据并在Flash reasoning条件异常时抛错，正常请求/回复原样返回。凭证在进程内从Keychain临时注入，不写入文件。

未自动重试、降并发、修改模型或实验机制。若provider分组RPM限制得到调整，可用相同配置/原始output目录**显式resume**，source/config/model/seed保持冻结；本轮结果归档保持只读。没有证据表明等待更久可以让P4持续工作，因此没有盲目resume。若更换并发或源码，必须新ID/new manifest，不能覆盖本run。
