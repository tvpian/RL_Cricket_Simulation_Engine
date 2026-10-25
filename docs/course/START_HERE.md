# Your hands-on reinforcement learning course

The aim is to understand why an agent chooses an action, how it learns from an
outcome, and whether that learning works outside its training conditions. You now
have **both a trainable batsman and a trainable bowler**. Start small; do not begin
by launching the longest experiment.

## Environment setup

In this prepared cloud workspace, activate the installed environment:

```bash
cd /workspace/RL_Cricket_Simulation_Engine
source /workspace/.venvs/cricket/bin/activate
```

For a fresh checkout on your own machine instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,dev,analysis]'
```

## 1. Play before training

From the repository root, activate your environment and run:

```bash
python -m cricket_rl.server
```

Choose **Batting** or **Bowling** under experiment setup, then **Start a new innings**.
For a shorter introduction, use **Your first decision** in the learning companion.
See the [playground guide](UI_GUIDE.md) for predictions, journals and snapshots. In batting mode the released ball
is shown before you choose stance, direction and power. Move fielders, vary power,
and replay deliveries. In bowling mode repeat a delivery and watch batting
anticipation change. Keep a log of ten decisions: what did you expect, what
happened, and how much of the outcome was random?

This is a tactical, delivery-level simulator. The batter sees the intended line
and length after release; it does not perceive video, estimate an unseen delivery,
or control a bat frame by frame. Noise, contact and future outcomes stay hidden.
The underlying physics and opponent response coefficients are uncalibrated.

## 2. Understand the five pieces of RL

| Piece | Bowling | Batting |
|---|---|---|
| Agent | Delivery selector | Shot selector |
| Observation | Batter, field, pitch, score and past deliveries (57 values) | Same context plus released delivery cue (66 values) |
| Action | Line × length: 9 choices | Stance × direction × power: 45 choices |
| Reward | `8 * wicket - total runs` | `total runs - 8 * wicket` |
| Episode | Innings until ball limit, wickets or target | Same innings rules |

Write down why maximizing runs alone could produce a very different batsman than
maximizing runs minus a wicket penalty. Rewards are design choices, not objective
truth. Illegal deliveries do not count as legal balls, and an unusual endless
sequence is reported as truncation rather than natural termination.

## 3. Run an agent that does not learn

```python
import gymnasium as gym
import cricket_rl  # registers both environments

env = gym.make('CricketBatting-v0', balls=6)
obs, info = env.reset(seed=42)
env.action_space.seed(42)
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(action, reward, info['runs'], info['wickets'])
    if terminated or truncated:
        break
env.close()
```

Repeat with `CricketBowling-v0`. Reset to the same seed and repeat the same action
sequence: the outcomes should match. Change just the action sequence: randomness
may be consumed differently, so equal seeds do not mean identical delivery noise.

**Deliverable:** explain the observation, action and reward for one delivery in
plain language, then compare random batting with a reactive baseline over at least
50 innings. A single impressive replay is not an evaluation.

## 4. Learn tabular Q-learning first

```bash
python -m cricket_rl.course_experiment --role batting --algorithm qlearning --steps 8192 --output runs/my-batting-q
python -m cricket_rl.course_experiment --role bowling --algorithm qlearning --steps 8192 --output runs/my-bowling-q
```

Read `QLearning.state` and the update in `course_experiment.py`. The table records
an estimate of future reward for each state/action combination:

`Q(s,a) ← Q(s,a) + α [reward + γ max Q(next_state, ·) − Q(s,a)]`

Here α=0.1 and γ=0.99; epsilon exploration decays to 0.05. Batting discretizes the
delivery cue and the largest field gap. Bowling discretizes expected delivery,
memory strength and batter preference. Many distinct observations map to one
state: this **state aliasing** is a deliberate limitation to inspect.

**Experiments:** change epsilon decay, learning rate, or the state encoding one at
a time. Plot reward against transitions from `learning.json`. Explain the
exploration/exploitation trade-off. Do not conclude that a bigger table is always
better: it also needs more data. The safety truncation is treated as a terminal
transition in this small teaching implementation.

## 5. Compare neural algorithms

Install the training extra, then run the same budgets:

```bash
pip install -e '.[train,dev]'
python -m cricket_rl.course_experiment --role batting --algorithm ppo --steps 8192 --output runs/my-batting-ppo
python -m cricket_rl.course_experiment --role batting --algorithm dqn --steps 8192 --output runs/my-batting-dqn
```

PPO adjusts a policy using fresh rollout batches. DQN learns action values using
stored transitions and a target network. Both use neural networks, but optimize
different objectives. Equal environment transitions do not imply equal compute
or equal gradient updates. Read their parameters in `course_experiment.py` and
the documentation for the installed Stable-Baselines3 version.

Repeat for bowling. Examine monitor CSVs, metadata, saved models, reward, runs,
wickets and action distributions. A neural agent can be worse than a simple rule.

## 6. Reproduce the course study

```bash
python -m cricket_rl.course_study --steps 8192 --episodes 50 --seeds 0 1 2 --output runs/course-study
```

This trains 18 models: two roles × three algorithms × three seeds. It retains all
seeds, compares random and role-specific heuristics, and evaluates both familiar
and held-out conditions. The held-out batting test combines different bowling,
pitch and batter attributes; it is a bundled stress test. To identify causes,
create separate experiments changing just one factor.

To generate the comparison figure and every seed's learning curve:

```bash
pip install -e '.[analysis]'
python -m cricket_rl.course_plot runs/course-study --output runs/course-plots
```

The final test seeds start at 300000. Tune on separate validation seeds, not those
test results. The supplied short pilot is a working reference, not an optimized
leaderboard. See [measured results](RESULTS.md).

**Deliverable:** a table of means and uncertainty across seeds, learning curves,
one failed policy, and an explanation of whether the result supports your hypothesis.

## 7. Play trained opponents

```bash
python -m cricket_rl.server --bowling-model runs/course-study/bowling-ppo-0/policy.zip --batting-model runs/course-study/batting-ppo-0/policy.zip
```

Select a role, start an innings, and choose **Trained policy**. You can instead
load a Q table (`policy.npz`) or a DQN model in the corresponding flag. The browser
selector dispatches the loaded algorithm; its legacy API policy key is `ppo`.

To train against a frozen learned opponent:

```bash
python -m cricket_rl.course_experiment --role batting --algorithm ppo --steps 8192 --opponent-model runs/course-study/bowling-ppo-0/policy.zip --output runs/bat-vs-frozen-bowler
python -m cricket_rl.course_experiment --role bowling --algorithm ppo --steps 8192 --opponent-model runs/course-study/batting-ppo-0/policy.zip --output runs/bowl-vs-frozen-batter
```

Keep opponent checkpoints fixed for evaluation. Changing both agents during
training introduces a moving opponent distribution. This project supports frozen
opponents; it does not yet implement simultaneous self-play, a league, or an Elo
rating. The course study exercises a fixed seed-0 PPO pair without picking test
winners.

## 8. Learn from failures

Useful next hypotheses:

- Does withholding the delivery cue make batting harder? Build a separate noisy
  perception variant rather than quietly changing the existing benchmark.
- Does a different wicket penalty change attacking versus defensive behavior?
- Does adding field coverage to tabular state improve transfer to new fields?
- Can training against several frozen bowlers outperform training against one?

Finish with a short report: problem, MDP, algorithms, experiment controls, results,
failures, limitations, and reproducible commands. This is a course project built
to teach RL; real-world cricket accuracy and SOTA performance remain unproven.
