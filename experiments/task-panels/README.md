# Activation task panel

`activation-smoke-inferai-deepseek-v4-flash.yaml` deliberately contains no E-task IDs.
The activation runner reads its ten E tasks from `activation-smoke-review.json` only after
the existing human task-review workflow has recorded eligibility and pairwise diversity for
those ten tasks and the configured V task. The runner refuses to load providers while that
review artifact is missing or incomplete. The activation protocol fixes E=10, V=93, H=0,
G=1, K=2, seed=1, and concurrency=4; test tasks are never scheduled.
