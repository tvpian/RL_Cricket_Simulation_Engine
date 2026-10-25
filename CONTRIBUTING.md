# Contributing to Cricket RL Lab

Start with the hands-on course in `docs/course/START_HERE.md`. Good contributions
include clearer lessons, reproducible experiments, deterministic rule scenarios,
bug fixes, accessibility improvements and calibrated simulation parameters.

- Keep UI, HTTP and Gym clients on the shared Python simulator. Do not create a
  second outcome generator merely to make a demo appear successful.
- Add meaningful tests for environment/rule/agent behavior. Run `pytest -q` and
  browser checks when changing web flows.
- Report algorithms, transition budgets, all seeds, evaluation splits and
  baselines. Keep failed runs. Distinguish training from inference and simulation
  reward from real-world accuracy.
- Keep generated checkpoints, caches and full experiment outputs in ignored
  `runs/`. Commit compact measured summaries and figures when appropriate.
- Do not add credentials, personal notes or downloaded datasets without checking
  their sharing permissions. Review exported journals before sharing them.
- Preserve the original material in `legacy/kivy/`. Document its provenance and
  distinguish it from new work.

No new license grant has been invented for the historical code/assets. Before
redistributing or reusing this work beyond the owner's intended release, clarify
licensing for original material and new contributions with the repository owner.
