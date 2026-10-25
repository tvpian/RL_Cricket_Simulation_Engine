# Cricket RL Lab

Learn reinforcement learning by **batting, bowling, experimenting and training**.
One shared cricket simulator powers two Gymnasium environments, a guided web
playground, and repeatable comparisons of Q-learning, PPO and DQN.

![Learning playground](docs/course/assets/ui-desktop.png)

## I just want to see it

**Browser demo:** [Open Cricket RL Lab](https://tvpian.github.io/RL_Cricket_Simulation_Engine/)

That link requires a successful GitHub Pages deployment. If it shows 404, enable
**Settings → Pages → GitHub Actions**, then run **Actions → Browser demo**.
The [Pages setup guide](web/README.md#github-pages) explains the exact steps.
The first visit may take a minute while Python loads in your browser.

For an immediate replay without installing anything, open [web/replay.html](web/replay.html),
use GitHub’s **Download raw file** button, and open the downloaded HTML file in
your browser. It contains a replay, not the full decision-making playground.

The browser demo runs real simulator steps using Python/NumPy in a Web Worker.
You can bat, bowl, edit the field and follow lessons without reaching a private
cloud IP. Training and PyTorch checkpoint inference use the local Python version.

## Run the full simulator on your computer

Requires Python 3.10+; Python 3.12 is the tested version.

```bash
git clone https://github.com/tvpian/RL_Cricket_Simulation_Engine.git
cd RL_Cricket_Simulation_Engine
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -e .
python -m cricket_rl.server
```

Now open **http://localhost:8000** in the browser **on that same computer**.
A private IP printed by a remote cloud workspace is not a public website URL.

Start **Your first decision** in the learning companion. Predict an outcome,
play a delivery, inspect **observation → action → reward → next state**, and
write a reflection. Then try repetition/adaptation or finding a field gap.

The playground includes manual and policy controls, camera/replay controls,
keyboard and drag field editing, reward breakdowns, local run snapshots,
downloadable journals, reduced-motion support and mobile layouts.
Playing or replaying a delivery does **not** train an agent.

## Learn and train

Follow the [hands-on course](docs/course/START_HERE.md), starting with random
actions and a small Q table before moving to neural algorithms.

```bash
# A first learning experiment; no PyTorch required
python -m cricket_rl.course_experiment --role batting --algorithm qlearning \
  --steps 8192 --output runs/my-first-batter

# Neural agents and development tools
pip install -e '.[train,dev]'
python -m cricket_rl.course_experiment --role bowling --algorithm ppo \
  --steps 8192 --output runs/my-first-bowler

# Both roles × Q-learning/PPO/DQN × three seeds
python -m cricket_rl.course_study --steps 8192 --episodes 50 \
  --seeds 0 1 2 --output runs/course-study
```

Outputs go into ignored `runs/` directories. Checkpoints are generated locally;
the repository does not ship newly trained models. After training, load them:

```bash
python -m cricket_rl.server \
  --batting-model runs/course-study/batting-ppo-0/policy.zip \
  --bowling-model runs/course-study/bowling-ppo-0/policy.zip
```

For Linux x86_64/Python 3.12, [requirements-tested.txt](requirements-tested.txt)
records the tested CPU training dependencies. Install its CPU Torch version from
`https://download.pytorch.org/whl/cpu` first. On other platforms, use the `train`
extra to resolve suitable wheels.

## What can the agents do?

| | Batting | Bowling |
|---|---|---|
| Choices | Stance × direction × power: 45 actions | Line × length: 9 actions |
| Observation | 66 values, including the released delivery cue and field | 57 values, including batting profile, field and history |
| Reward | Runs − 8 × wickets | 8 × wickets − runs |
| Opponent | Scripted or frozen learned bowler | Adaptive scripted or frozen learned batter |

Batting is a **delivery-level tactical task**: the intended line/length cue is
visible after release. It is not video perception or frame-by-frame bat control.
Both roles share flight, bounce, contact, fielding and innings rules. Frozen
learned opponents are supported; simultaneous self-play is not implemented.

```python
import gymnasium as gym
import cricket_rl

env = gym.make('CricketBatting-v0', balls=6)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(22)
env.close()
```

The server also exposes isolated Gym-style create/reset/step/close sessions.
See the [API reference](docs/course/API.md) and [remote Gym adapter](cricket_rl/remote.py).
The local playground is a single-user development app, not an authenticated
multi-user hosted service.

## Results, honestly

The two-role pilot completed 18 runs and 147,456 transitions. Batting learners
improved mean reward over the reactive baseline in familiar conditions; the
bowling learners still lost to the varying heuristic. Results include failed
seeds and uncertainty rather than selecting the best replay.

- [Measured course results and learning curves](docs/course/RESULTS.md)
- [Current playground guide](docs/course/UI_GUIDE.md)
- [Simulator architecture and limitations](docs/research/REPORT.md)
- [Earlier adaptive-batting experiment](docs/research/ADAPTIVE_RESULTS.md)

The physics and opponent parameters are **uncalibrated**. This is a learning and
research platform, not a validated predictor of real cricket or a SOTA claim.

## Repository map

```text
cricket_rl/          Shared simulator, Gym environments, agents and web server
  dashboard.*       Learning playground UI
  course_*.py        Training, comparison and plotting exercises
web/                Browser Python worker and static-hosting instructions
tools/              Build tools for the GitHub Pages site
tests/              Physics, rules, environments, HTTP and browser tests
docs/course/        Hands-on course, screenshots, APIs and measured results
docs/research/      Archived studies and technical reports
legacy/kivy/        Original scripts, assets, video and neural checkpoint
.github/workflows/  Tests and browser-demo deployment
```

The [original project](legacy/README.md) is preserved byte-for-byte under
`legacy/kivy/`. Its original description is available there. The new course
agents are separate implementations; the original saved agent's training has
not been recovered or reproduced.

## Develop and contribute

```bash
pip install -e '.[dev]'
pytest -q

# Actual-browser checks
pip install -e '.[browser]'
python -m playwright install chromium
CRICKET_BROWSER_TEST=1 pytest tests/test_browser.py -q

# Plot your study
pip install -e '.[analysis]'
python -m cricket_rl.course_plot runs/course-study --output runs/course-plots
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) for experiment and review expectations.
For an HTML replay that opens directly as a downloaded file:

```bash
python -m cricket_rl.replay --policy random --deliveries 12 --output runs/replay.html
```

The complete browser playground needs HTTP hosting to load its bundled Python
runtime. It does not fetch third-party packages in the browser. The exported replay is a separate self-contained artifact.
