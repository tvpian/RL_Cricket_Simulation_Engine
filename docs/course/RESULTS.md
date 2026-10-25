# Two-role course experiment

Budget: 18 runs × 8192 requested transitions. 50 evaluation innings per scenario; seeds [0, 1, 2]. Test seeds begin at 300000. All seeds retained. This is a short pipeline pilot, not evidence of convergence.

| Role | Algorithm | Scenario | Mean reward ± seed interval | Runs/innings | Wickets/innings |
|---|---|---|---:|---:|---:|
| batting | qlearning | familiar | 56.63 ± 15.29 | 68.69 | 1.51 |
| batting | qlearning | heldout | 40.87 ± 25.59 | 61.45 | 2.57 |
| batting | ppo | familiar | 60.61 ± 11.75 | 71.11 | 1.31 |
| batting | ppo | heldout | 49.90 ± 40.22 | 69.53 | 2.45 |
| batting | dqn | familiar | 63.27 ± 3.82 | 73.67 | 1.30 |
| batting | dqn | heldout | 57.15 ± 12.27 | 76.08 | 2.37 |
| bowling | qlearning | familiar | -40.91 ± 21.23 | 54.25 | 1.67 |
| bowling | qlearning | heldout | -56.31 ± 26.59 | 68.36 | 1.51 |
| bowling | ppo | familiar | -68.18 ± 0.00 | 82.26 | 1.76 |
| bowling | ppo | heldout | -94.12 ± 0.00 | 104.20 | 1.26 |
| bowling | dqn | familiar | -41.05 ± 11.61 | 50.01 | 1.12 |
| bowling | dqn | heldout | -57.85 ± 10.58 | 62.92 | 0.63 |

Baselines (episode means; no training-seed interval):

| Role | Baseline | Familiar reward | Held-out reward |
|---|---|---:|---:|
| batting | random | 31.00 | 26.34 |
| batting | reactive | 42.46 | 38.78 |
| bowling | random | -39.52 | -59.80 |
| bowling | varying | -32.18 | -38.36 |

Reward is runs − 8 × wickets for batting, and its negative for bowling. Roles have different opponents and decision timing; their scores are not a ranking between roles. Held-out combines expert attributes, dry pitch, and (for batting) varying bowling. This is a bundled stress test, not an isolated causal ablation.

Intervals are Student-t intervals across training-seed means, not a formal superiority test. The simulator remains uncalibrated. Frozen PPO seed-0 opponents were also exercised in both directions; see study.json. These matches are sequential fixed-policy evaluations, not simultaneous self-play or a win-rate league.

## Audit revision and interpretation

These results were re-run after correcting Q-learning's truncation bootstrap and
seeding protocol. The prior version is preserved in
[the pre-audit report](../research/COURSE_V1_RESULTS.md). This is a regression
re-evaluation on the same already-published test seeds, not a new blind test or
proof of convergence. No test-based checkpoint selection was performed.

All 18 jobs completed 8,192 transitions each (147,456 total); no learned-policy
evaluation innings were truncated. The frozen PPO pair gave identical innings
and opposite role rewards in all 50 matched seeds. Both baselines and all training
seeds are retained. Intervals over three seed means are wide; do not use a point
estimate alone as a superiority claim.

The browser's small Q-learning demo tables use predetermined seed 0, not the best
test model. Their [model card](../../web/models/model-card.json) documents training,
checksums and limitations. Bowling learners still do not beat the varying
heuristic on average. This is a useful negative result, not a bug to conceal.

![Algorithm comparison](assets/comparison.svg)

![Every training seed](assets/learning.svg)

Full outcomes and checkpoints are generated under `runs/audit-course-study/`.
[Machine-readable summary](summary.json) · [Hands-on lessons](START_HERE.md)
