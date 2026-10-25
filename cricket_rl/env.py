"""Gymnasium interface for the trajectory-based cricket strategy simulator."""
import numpy as np
from cricket_rl.gym_backend import gym, spaces
from cricket_rl.physics import PITCHES, Pitch, simulate
from cricket_rl.rules import Innings
from cricket_rl.batting import BowlingMemory, BattingPlan

STYLES = {
    "defensive": (0.25, 0.45),
    "balanced": (0.55, 0.60),
    "aggressive": (0.85, 0.65),
    "unseen": (0.70, 0.85),
}
OBSERVATION_VERSION = 3
OBSERVATION_SIZE = 57


def default_field(rng, boundary):
    """Nine outfielders; bowler and keeper are not explicitly simulated."""
    angles = np.array([-2.7, -1.8, -.95, -.5, 0, .5, .95, 1.8, 2.7])
    angles += rng.normal(0, .12, 9)
    radii = rng.uniform(12, min(55, boundary - 3), 9)
    return np.column_stack((np.sin(angles) * radii, np.cos(angles) * radii))


class CricketEnv(gym.Env):
    """9 bowling actions, 57 normalized observations; reward=8*wickets-runs.

    Observation v3 includes full fielder positions, pitch conditions, free-hit
    state, handedness, chase target and score. Innings end naturally; a guard
    against endless illegal deliveries is reported as truncation.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, styles=("defensive", "balanced", "aggressive"), balls=24,
                 max_wickets=3, render_mode=None, pitch="standard", target=None,
                 wicket_reward=8.0, record=False, randomize_pitch=False, adaptive_batting=True, batting_policy=None):
        super().__init__()
        if not styles or any(s not in STYLES for s in styles):
            raise ValueError("Provide known batting styles")
        if not isinstance(balls, int) or not 1 <= balls <= 300:
            raise ValueError("Ball limit must be an integer in [1, 300]")
        if not isinstance(max_wickets, int) or not 1 <= max_wickets <= 10:
            raise ValueError("Wicket limit must be in [1, 10]")
        if target is not None and (not isinstance(target, int) or not 1 <= target <= 500):
            raise ValueError("Target must be an integer in [1, 500]")
        if render_mode not in (None, "ansi"):
            raise ValueError("Only ansi rendering is supported")
        if isinstance(pitch, str) and pitch not in PITCHES:
            raise ValueError("Unknown pitch")
        if not isinstance(pitch, (str, Pitch)) or not np.isfinite(wicket_reward):
            raise ValueError("Invalid configuration")
        self.styles, self.balls, self.max_wickets = tuple(styles), balls, max_wickets
        self.render_mode, self.pitch_config, self.target = render_mode, pitch, target
        self.wicket_reward, self.record = float(wicket_reward), record
        if not isinstance(adaptive_batting, bool):
            raise ValueError("adaptive_batting must be a boolean")
        self.batting_policy = batting_policy
        self.adaptive_batting = adaptive_batting
        self.randomize_pitch = randomize_pitch
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(0, 1, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        style = options.get("style")
        if style is not None and style not in STYLES:
            raise ValueError("Unknown batting style")
        self.pitch = (PITCHES[str(self.np_random.choice(tuple(PITCHES)))]
                      if self.randomize_pitch else
                      PITCHES[self.pitch_config] if isinstance(self.pitch_config, str) else self.pitch_config)
        self.lineup = []
        for _ in range(11):
            batting_style = style or str(self.np_random.choice(self.styles))
            self.lineup.append((batting_style, int(self.np_random.integers(3)),
                                int(self.np_random.choice([-1, 1]))))
        self.memories = [BowlingMemory() for _ in range(11)]
        self.innings = Innings(self.balls, self.max_wickets, self.target)
        self.fielders = default_field(self.np_random, self.pitch.boundary)
        if "fielders" in options:
            self.set_field(options["fielders"])
        self.last_delivery = None
        self.done = False
        return self._observation(), self._info()

    def set_field(self, fielders):
        positions = np.asarray(fielders, dtype=float)
        if positions.shape != (9, 2) or not np.isfinite(positions).all():
            raise ValueError("Provide nine finite 2D positions")
        if np.any(np.linalg.norm(positions, axis=1) >= self.pitch.boundary):
            raise ValueError("Fielders must be inside the boundary")
        self.fielders = positions.copy()

    @property
    def style(self):
        return self.lineup[self.innings.striker][0]

    @property
    def preference(self):
        return self.lineup[self.innings.striker][1]

    @property
    def handedness(self):
        return self.lineup[self.innings.striker][2]

    @property
    def aggression(self):
        return STYLES[self.style][0]

    @property
    def skill(self):
        return STYLES[self.style][1]

    @property
    def field(self):
        angles = np.arctan2(self.fielders[:, 0], self.fielders[:, 1])
        return np.array([np.mean(np.abs(angles - a * self.handedness) < .5)
                         for a in [-.9, 0, .9]])

    @property
    def runs(self):
        return self.innings.runs

    @property
    def wickets(self):
        return self.innings.wickets

    @property
    def deliveries(self):
        return self.innings.balls

    @property
    def batting_plan(self):
        return self.memories[self.innings.striker].plan() if self.adaptive_batting else BattingPlan()

    def _observation(self):
        p = self.pitch
        conditions = [p.restitution, p.friction, p.boundary / 90,
                      min(p.swing / 5, 1), p.pace / 45,
                      min(p.accuracy, 1), p.no_ball_rate / .2]
        return np.array([self.deliveries / self.balls, self.wickets / self.max_wickets,
                         self.aggression, self.skill, self.preference / 2,
                         *self.field, *((self.fielders.ravel() + 90) / 180),
                         *conditions, float(self.innings.free_hit), (self.handedness + 1) / 2,
                         (self.target or 0) / 500, min(self.runs / 500, 1),
                         *self.memories[self.innings.striker].observation(),
                         float(self.adaptive_batting)], dtype=np.float32)

    def _info(self):
        plan = self.batting_plan
        return self.innings.as_dict() | {"style": self.style, "observation_version": OBSERVATION_VERSION,
                                         "adaptive_batting": self.adaptive_batting,
                                         "expected_action": plan.expected_action,
                                         "anticipation": plan.confidence}

    def step(self, action):
        if self.batting_policy is not None and not self.done:
            from cricket_rl.roles import batting_observation
            from cricket_rl.shots import Shot
            if not self.action_space.contains(action):
                raise ValueError("Action must be an integer in [0, 8]")
            shot_action = int(self.batting_policy.predict(batting_observation(self, action), deterministic=True)[0])
            return self.execute(action, shot=Shot.decode(shot_action))
        return self.execute(action)

    def execute(self, action, shot=None):
        if self.done:
            raise RuntimeError("Call reset before stepping a finished episode")
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer in [0, 8]")
        faced_by = self.innings.striker
        self.last_delivery = simulate(self.np_random, *divmod(int(action), 3),
                                      self.aggression, self.skill, self.preference,
                                      self.fielders, self.pitch, record=self.record,
                                      handedness=self.handedness, free_hit=self.innings.free_hit,
                                      batting_plan=self.batting_plan, shot=shot)
        d = self.last_delivery
        self.memories[faced_by].observe(int(action))
        self.innings.apply(d)
        terminated = self.innings.complete
        truncated = not terminated and self.innings.attempts >= self.balls * 4
        self.done = terminated or truncated
        info = self._info() | {"delivery_runs": d.runs + d.extras, "wicket": d.wicket,
                               "dismissal": d.dismissal, "event": d.event,
                               "extra_type": d.extra_type, "legal": d.legal,
                               "delivery_anticipation": d.anticipation,
                               "readiness": d.readiness, "surprise": d.surprise}
        reward = self.wicket_reward * d.wicket - d.runs - d.extras
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        return f"{self.runs}/{self.wickets} after {self.innings.balls // 6}.{self.innings.balls % 6} overs"
