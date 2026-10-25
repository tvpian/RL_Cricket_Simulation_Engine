# Cricket Strategy Lab: technical report

## Question and contribution

Does training a bowling policy against diverse batting styles improve transfer to
an unseen batting lineup and unseen pitch surfaces, compared with single-style
training? The project supplies an inspectable simulator, an independent cricket
scorekeeping state machine, an RL interface, an interactive demonstration, and a
reproducible multi-seed experiment to investigate that question.

The contribution is an experimental platform and its measured findings. Neither
real-world accuracy nor superiority over published cricket systems is established.
The original static experiment is preserved in [RESULTS.md](RESULTS.md). The
current adaptive-batting results and completed budget are in
[ADAPTIVE_RESULTS.md](ADAPTIVE_RESULTS.md).

## Architecture

The dependency direction is `physics → rules → environment → experiments/UI`.
The rules consume a delivery result rather than depending on graphics or a trained
policy. The HTTP server calls the same Gymnasium environment used for training.
The browser renders server-produced trajectories; it does not implement a second
physics or scoring model. Disabling recording changes memory use, not outcomes.
The original Kivy prototype is retained as historical material.

The server is a single-session local development tool. It binds to loopback by
default, rejects cross-origin writes, limits request sizes, and has no user
accounts or persistent shared storage. Do not deploy it as a public service.

## Coordinates, time and ball dynamics

Coordinates are metres relative to the striker: x is lateral, y points toward the
bowler, and z is height. The full pitch is drawn as 20.12 m; release is 18 m from
the striker at 2 m height. The fixed integration step is 0.02 s (50 Hz).

Before impact, velocity-Verlet integration applies gravity and constant sampled
lateral acceleration. For acceleration a, each step uses
`position += velocity * dt + 0.5 * a * dt²` and `velocity += a * dt`.
Bowling speed is sampled around 34 m/s and limited to a physical operating range.
Line and length choose a lateral target and nominal bounce location. The initial
vertical velocity is solved for that bounce location under gravity.

On pitch impact, vertical speed is multiplied by restitution, horizontal speed
by one minus pitch friction, and a penetration correction returns the ball above
ground. This is an impact model, not a deformable ball/surface simulation.
After bat contact, loft and placement determine the outgoing velocity. Linear
horizontal drag acts in flight and constant deceleration acts on the ground.
The standard, green, and dry presets change bounce, friction, and swing. These
coefficients are illustrative hypotheses, not fitted measurements.

Bat contact uses a probability conditioned on skill, aggression, target mismatch
and impact height. Timing determines shot speed; poor timing can generate edges
behind the wicket. Contact is not a rigid-body bat collision. Handedness mirrors
targets and shot preferences. A lineup contains eleven seeded batting profiles.

## Adaptive batting and observable history

Each batter has nine exponentially decayed delivery weights (retention 0.75 per
faced delivery), a 0.02 prior per action, and a last-action code including an
unfaced sentinel. The next expected delivery is the most likely line/length pair.
Confidence is the predicted probability above the uniform 1/9 baseline, rescaled
to [0, 1]. The plan is computed **before** the current action is observed and the
memory updates only after that delivery. Other batters have separate memories;
new batters have no history and memories reset each innings.

Preparation blends the batter's preferred position with the expected line. After
the delivery is revealed, readiness scales with how closely its line and length
match the forecast; surprise is the remaining confidence. Readiness increases
contact and timing quality, reduces timing noise, and permits a premeditated shot
toward a field gap. Surprise has the opposite effect on contact and timing.
These are explicit hypothetical opponent dynamics, not fitted cricket behavior.
No repetition bonus or penalty is added to the bowler reward. A static-batting
switch disables these effects while retaining the same observation layout.

All 19 memory values and the adaptation flag are exposed to the bowling policy,
so it can condition actions on what the current batter has faced. The current
batter's expectation and last-delivery readiness are also shown in the dashboard.
The static control is an evaluation ablation; a policy trained only against
adaptive opponents may not generalize to its disabled-adaptation flag.

## Fielding and running

Nine outfielders have explicit positions. After a 0.3 s reaction delay, every
fielder pursues the current ground projection at up to 6 m/s. Movement is bounded
and cannot overshoot the target. A fielder within 1.2 m can gather a ground ball or
attempt a catch below 2.3 m; catch success is 85%. A dropped catch cannot be retried
by that same fielder until the ball grounds. Bowler and wicketkeeper are drawn or
implicit, not simulated as additional catching players.

Running uses 3.5 s for the first run and approximately 3 s for later runs, with a
28 m/s return throw and 0.5 s gathering allowance. Marginal attempts can produce
run-outs. Runs are capped at three except boundaries; this excludes overthrows
and unusual all-run fours. These timings are heuristic and need calibration.
Fielders reset to their tactical starting positions between deliveries.

## Rules and deliberate approximations

The independent innings model counts legal balls, wides, no-balls, total runs,
extras, wickets, and strike. Odd completed running runs switch ends; each legal
over switches ends again. Wickets introduce a new batter, and innings finish at
the legal-ball limit, wicket limit, or successful chase.

A foot-fault no-ball is sampled; an untouched ball outside a simplified lateral
or height envelope is a wide. No-ball takes precedence. A no-ball grants a free
hit, and a subsequent wide preserves it. Bowled, LBW, and caught dismissals are
suppressed on no-balls/free hits; run-outs remain possible. A protected catch is
kept live by grounding the ball and continuing fielding. No-ball penalty plus bat
runs are both scored. Boundary classification depends on whether the struck ball
has grounded before crossing the circular boundary.

The LBW model is explicitly approximate: a straight missed ball below bail height
may be treated as intercepted by a pad. It does not check full pitching/impact
criteria or reconstruct umpire review. Run-outs assume which runner is dismissed
rather than simulating both runners. Wides do not model attempted stumpings,
byes/leg-byes and penalty runs are absent, and height-based no-ball rules and
bouncer limits are not implemented. Field-placement restrictions, powerplays,
bowler over limits, player fatigue, weather, spin aerodynamics, seam orientation,
and multi-innings match logic are also absent. This is not full law compliance.

## MDP and experimental controls

The action is one of nine line/length choices. Observation version 3 has 57
normalized values: legal-ball and wicket progress, current batter aggression and
skill, preferred line, three coverage summaries, all nine fielder coordinates,
seven pitch parameters, free-hit flag, handedness, chase target, and total score. Nineteen bowling-memory values plus the adaptation flag extend
the earlier 37-value layout. Old checkpoints are rejected with a retraining message.
Innings defaults are 24 legal balls and three wickets. Long illegal-delivery
sequences have a separate truncation guard and are not counted as normal endings.

Reward is `8 * wickets − runs conceded`, including extras. Eight is a chosen
utility, not an estimate of wicket value. Runs per over, wickets per innings, raw
runs, balls, and reward are all retained to expose trade-offs. A lower economy rate
alone does not establish a better bowling policy; early all-outs change exposure.

The full future batting lineup is hidden, making this a partially observable
control problem despite exposing the current batter. The opponent adapts through an explicit memory model,
not through a learned adversarial policy. The project does not solve multi-agent RL.

PPO uses an MLP policy, 256-step rollouts, batch size 64, CPU, and the remaining
Stable-Baselines3 defaults of the recorded package version. Single-style agents
see balanced batters; mixed-style agents see defensive, balanced, and aggressive
batters. Both train on the standard pitch and share the same budgets and seeds.
Five training seeds are used. Evaluation runs use separate seeds starting at
100000, in four adaptive scenarios: familiar batting mixture, unseen expert lineup,
unseen lineup on green, and unseen lineup on dry, plus a fifth scenario disabling
adaptation for the unseen lineup on the standard pitch.

Random, field-aware good-length, and history-aware varying-length heuristics are evaluated on the same initial
seeds. Action choices change random-number consumption, so noise is not identical
between policies. The held-out expert differs in numeric skill/aggression, not an
independently authored batting algorithm; it is a limited generalization test.

## Reporting and reproducibility

Each training run stores its requested and actual step budget, seed, elapsed time,
PPO configuration, dependency versions, source SHA256 hashes, a model checkpoint,
and a monitor CSV. Evaluation stores episode-level outcomes and truncation flags.
Source hashes make uncommitted code auditable; a commit hash alone cannot do so.

Per-policy episode summaries use normal approximate 95% confidence intervals.
The multi-seed report uses Student-t intervals over five independent training-seed
mean rewards. Baseline intervals instead describe evaluation-episode uncertainty;
these should not be compared as if they came from the same sampling process.
Zero training-seed intervals can arise when all learned policies choose the same
action; this does not eliminate uncertainty from the finite evaluation set.
The constant-strategy baseline selects one of the nine fixed actions using
100 validation episodes per action at seeds 50000–50099 on familiar batters,
then evaluates that selected action at the separate test seeds. Action histograms
reveal whether a learned policy adapts or collapses to one delivery.
The fixed evaluation scenarios are shared across seeds and cannot establish
performance across all possible opponents. No test-set model selection is used.
The finite training budget is a pilot; convergence is not assumed.

A separate matched-state diagnostic changes only the current batter's memory
between six yorkers and six good-length deliveries in 30 synthetic observations.
Field positions, score, pitch and batter attributes are identical. A changed
policy decision demonstrates sensitivity to observed history, not superior
playing performance. These synthetic probes are not part of the evaluation set
and do not select models or tune coefficients.

## Validation and next scientific step

Unit and integration tests check Gymnasium compliance, seed reproducibility,
trajectory finiteness and nonnegative height, pursuit speed bounds, scoring
identities, extras and free-hit edge cases, strike rotation, targets, all-out,
input validation, actual HTTP operations, and PPO checkpoint round-trips.
Adaptive tests verify memory updates, forgetting, per-batter isolation, no current-
action leakage, static controls, and fixed-seed effects on batting contact. An
actual-browser test checks controls, editable fields, playback, downloads,
JavaScript errors and narrow-screen layout. These establish software behavior,
not empirical predictive accuracy.

The next scientific step is data calibration: obtain permitted ball tracking and
outcome data, define a training/validation split, fit bounce and shot distributions,
and compare held-out pace, bounce height, scoring, wicket and boundary distributions.
Until that is done, present the project as a reproducible simulation/decision-making
study, not a predictor of professional cricket outcomes. Further full-law features
and competitive search or recurrent baselines are natural extensions.
