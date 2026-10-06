# Partial mechanism-smoke result

This is an incomplete real-provider run, archived for diagnosis and analysis. It is not a completed alternating-evolution experiment and is not research evidence for Service repair.

- The native incumbent panel completed at **16/20 = 0.80**.
- GPT-6.1-Sol produced one Customer candidate. Its E accuracy was **17/20 = 0.85**, so the lower-accuracy selection rule retained the incumbent Customer.
- The generation-0 Service Evolver did not produce a proposal. Two 60-second client read timeouts were followed by one HTTP 504 from the upstream CDN after extending the client read timeout to 180 seconds. No model switch or additional retry was made.
- Therefore Service E accuracy, Service acceptance, generation 1, V, and H results are unavailable. V/H content was not loaded.
- The timeout180 run reuses 40 completed native episodes (20 incumbent, 20 candidate) from the earlier run. Five incomplete attempts remain in `episodes/` as diagnostics.
- Combined wall-clock from the first launch through the recorded stop was approximately **93.8 minutes**.

Provider usage across the attempts is in `api-usage-live.json`; each completed trajectory and `EpisodeRecord` is under `episodes/`. The 60-second superseded run remains in the local ignored `experiments/runs/` directory; this archive contains the canonical timeout180 manifest and reused trajectories.
