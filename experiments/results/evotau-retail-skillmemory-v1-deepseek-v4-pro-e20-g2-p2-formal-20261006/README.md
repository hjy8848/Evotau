# Retail E20/G2/P2 — InferAI group RPM failure

**Status: failed / 未完成。并发2比并发4多完成了9个episode，但同样遇到分组RPM限制；没有正式accuracy或evolution结果。**

| 条件 | 完整episode | Accuracy | 状态 |
|---|---:|---|---|
| Gen0 C0 × S0 | 9/20 | N/A，panel未完成 | HTTP429停止 |
| Gen0 Customer proposal/selection | 未启动 | N/A | 没有candidate或paired transitions |
| Gen0 Service ADD/UPDATE/NO_OP | 未启动 | N/A | 没有proposal或candidate replay |
| Gen1 | 未启动 | N/A | 未到达，无cache reuse统计 |

## Frozen inputs / validity

- experiment ID: `evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g2-p2-formal-20261006`
- source commit: `51bbe03cc4259555e3a53b74a0da91553b15e434`
- source SHA256: `965497fcfd0260fdf20cc2525fecc024d8853bd82d1baa44730f1b0ce3902f3e`（与cda8ca8 release canary一致，源码未改）
- manifest SHA256: `c1505a050c708fe7f288d3c61bd6659b46f583d01e2b946ada1de85e1f4e8679`
- config SHA256: `31d18a669a8594869747eabc07c7c52e156fe92571738a77c9c73b43de2183ba`
- Only concurrency changes P4→P2 plus independent identity/output/checkpoint. Frozen Retail/E20/G2/candidate1/seed1/fitness seed1/max_steps32/unlimited request budget/model args/prompts/carriers/selection unchanged.
- E20: `66 92 29 67 106 22 69 98 93 88 4 21 8 54 107 48 52 80 35 16`；全部是合法train IDs，V/H/excluded互斥。V/H内容未加载，run flags均false。初始SkillMemory=[]，没有新的active memory，skill count trajectory只有初始0。
- 114 tests通过，ruff/src/tests与diff检查通过。独立新run未复用P4/canary/旧run output、checkpoint或episode cache。
- resume_count=0。没有生成完整generation/checkpoint/alternating-result，未伪造accuracy或completed_generation。

## Provider failure / evidence

HTTP200=192，HTTP429=3，timeout=0，5xx=0。真实错误：`group requests-per-minute limit exceeded` (`rate_limit_exceeded`)。具体RPM quota/reset时间没有由provider错误独立证实。并发2未消除每分钟请求次数限制：限制并发数并不等于保证每分钟请求速率；后续需要处理provider RPM配额或另行明确请求节流，不能只根据这次尝试承诺再减并发即可成功。

完整task IDs: `66 92 29 67 106 22 69 98 93`。失败attempt task IDs: `88 4 21`。已有9份有效native score/trajectory保存；未完成条件未计failure/fitness、未跳task继续selection。各失败attempt保存partial trajectory、异常链、failure stage、usage和provider诊断。累计3个失败attempt不意味着3路并发：有queued job在worker释放后、取消生效前开始；观察到的并发峰值是2。

Wire request/response与预算/provider logs均为195条，response IDs唯一归属episode，exact per-episode attempts/tokens与全局对账。num_retries全部0，未观察到隐藏transport retries；in_flight=reserved=0。9条completed panel references；12个实际开始的episode jobs（9完整+3未完整），未开始的8个queued jobs取消；合法cache hits=0。本轮native `EpisodeJobTelemetry`在panel失败前未最终落盘，未伪造peak字段；wire实测不同worker同时请求峰值2，单worker内部请求串行，executor上限2。

195/195 Flash请求实际为`thinking.type=disabled`、temperature0。192个成功回复reasoning_tokens全0，没有reasoning文本，没有异常空completion；3个429错误体不属于模型空回复，失败usage不可用。Pro/Evolvers=0；high effort仍只是配置请求，未得到provider验证。Reviewer/Customer Judge/Service Judge=0。

## API / time

| Role | Calls | Success | Failure | Prompt | Completion | Sum seconds | Mean seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| customer | 75 | 73 | 2 | 68,692 | 2,706 | 234.51 | 3.13 |
| service | 117 | 116 | 1 | 760,863 | 16,351 | 536.83 | 4.59 |
| evaluator | 3 | 3 | 0 | 13,120 | 276 | 15.25 | 5.08 |
| customer_evolver | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 |
| service_evolver | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 |
| reviewer | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 |
| customer_judge | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 |
| service_judge | 0 | 0 | 0 | 0 | 0 | 0.00 | 0.00 |

Total: **195 calls；842,675 prompt tokens；19,333 completion tokens；862,008 total tokens；wall-clock 398.38秒（6分38秒）；sum API time 786.59秒**。token统计只含192个有usage的成功响应。Flash=195，Pro=0，unique complete episodes=9，panel refs=9，reuse/cache=0。没有完成evolution，无法拆分完整phase时间或计算speedup。P4那次38.82秒/0 complete，P2这次398.38秒/9 complete，以及P1 canary795.33秒/6 complete的工作量/条件不同，不能直接算严格加速比。

## Mechanism interpretation and resume

Customer challenge、Service reusable skill、ADD/UPDATE/NO_OP credit assignment、repaired/regressed IDs、paired transitions和challenge→repair dynamics均未发生。partial panel的7 pass/2 fail是各自有效native task结果，不代表完整E20 accuracy；其余11个task的score仍未知。不能据此归因Evolver/Customer/Skill representation能力或得出泛化结论。

没有自动重试、降为1并发或模型替换。原始run目录保留；未来在provider RPM问题得到处理后，可用相同冻结P2 config/source/model/seed显式resume，复用9个完整condition，只补未完成/未开始条件。归档为此次attempt只读快照。若修改并发/源码/请求策略，需要明确新条件和manifest，不能混作本次实验。

- `formal-attempt-summary.json` contains full machine-readable manifest, role usage, validity checks and failed attempts.
- `episodes/` preserves 9 completed native simulations/records and 3 failed partial trajectories/diagnostics.
- `actual-provider-http.jsonl` preserves sanitized actual wire metadata; no keys/headers/messages/reasoning text.
- `config.yaml`, `preflight.json`, `manifest.json`, `run-context.json`, `run-execution-state.json` preserve frozen inputs and failure lifecycle.
- `execution-observer.py` is the actual invocation script, using Keychain only inside the process and passing normal requests/replies unchanged.
