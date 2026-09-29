# EvoTau DeepSeek V4 Flash evolution smoke

This archive contains the first completed, provider-backed two-generation EvoTau Core smoke on pinned τ-bench Retail task 73. The final Phase 3 run completed both generation commits, five native E episodes, and no incomplete episodes. The Phase 0 parent reached native reward 1.0 and was completed by replaying only the native τ-bench review on its already saved episode after a transient provider failure.

## Result

All five Phase 3 episodes received native reward 1.0. The role-separated Customer audit marked all five Customers valid and strategy-adherent, with no policy violation in the frozen MVP taxonomy. The Customer Evolver proposed legal mutations, but neither generation strictly improved fitness, so the incumbent Customer and Service strategies were retained. The second generation had only one unseen legal mutation available after excluding strategies already evaluated; this is why the run has five rather than six episodes. No verified failure was available, so the Service repair gate was correctly skipped.

The configured V task 93 did not run: the frozen controller uses V only for a Service repair gate, and there was no verified failure to trigger that gate. No heldout task was selected or run.

The Phase 3 run is an integration smoke, not evidence of a measured evolutionary advantage. The audit used DeepSeek V4 Flash in separate audit prompts and has not been calibrated against human annotations. The native τ-bench qualitative reviewer returned an empty response that failed JSON parsing in all five Phase 3 episodes; τ-bench reward evaluation and EvoTau's independent role audit completed, and the result records each reviewer parse error. Review-dependent analyses should wait for a reviewer-output fix and calibration.

## Provider and budget

- Model: `openai/deepseek-v4-flash` via `https://inferaiapi.com/v1`; thinking disabled for every frozen role.
- Final Phase 3 cap: 164 provider attempts, with 99 used including the 17-attempt Phase 0 parent; 82 attempts were used by the successful Phase 3 attempt itself.
- A prior Phase 3 attempt used 16 attempts before the audit parser rejected its output. The retry cap was reduced to preserve the original 180-attempt cumulative Phase 3 budget ceiling across both attempts.
- Total provider attempts across parent setup and both Phase 3 attempts: 120, including two failed provider calls and one parser-rejected episode. Prompt/completion usage totals are in `analysis-summary.json`. Dollar cost is unavailable because the installed LiteLLM price table does not recognize this model alias.

Phase 0 attempt 1 stopped on a Cloudflare 520 before saving a simulation. Attempt 2 completed the task (reward 1.0) but failed during native review; the recovery parent records the source result and trajectory hashes and re-ran only full review/auth classification. The recovery result is the parent used by Phase 3.

## Contents

- `configs/`: exact frozen configurations for every attempt.
- `runs/phase0-completed-parent/`: complete parent manifest, reviewed trajectory, recovery lineage, and result.
- `runs/phase3-attempt2/`: complete Phase 3 result, generation commits, per-episode trajectory/record/telemetry, provider provenance, and archive database.
- `runs/phase3-attempt1/`: parser-rejected episode and saved native trajectory.
- Other `runs/phase0-*` directories preserve the failed provider attempts and their accounting.
- `checkpoints/phase3-attempt2-checkpoint.json`: final generation checkpoint.
- `SHA256SUMS`: integrity hashes for all files in this archive.

The callback implementation is in `src/evotau/provider_plugins/deepseek_v4_flash.py`; the submitted source/config revision is commit `c7738c4`. No API key or temporary provider log is included.
