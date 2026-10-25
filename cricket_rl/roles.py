"""Delivery-level batting Gymnasium task sharing bowling physics and rules."""
import numpy as np
from cricket_rl.gym_backend import gym, spaces
from cricket_rl.env import CricketEnv, OBSERVATION_SIZE
from cricket_rl.shots import Shot, DIRECTIONS

BATTING_OBSERVATION_SIZE = OBSERVATION_SIZE + 9


def batting_observation(base, delivery):
    cue = np.zeros(9, dtype=np.float32)
    if delivery is not None:
        cue[delivery] = 1
    return np.r_[base._observation(), cue].astype(np.float32)


def batting_baseline(env, policy, rng):
    if policy == 'random':
        return int(rng.integers(45))
    if policy != 'reactive':
        raise ValueError('Unknown batting baseline')
    stance = env.pending_delivery // 3
    angles = np.arctan2(env.base.fielders[:, 0], env.base.fielders[:, 1])
    candidates = np.array(DIRECTIONS) * env.base.handedness
    coverage = np.exp(-((angles[:, None] - candidates[None, :]) / .3)**2).sum(axis=0)
    return Shot(stance, int(np.argmin(coverage)), 1).encode()


class BattingEnv(gym.Env):
    """45 stance/direction/power choices; observe the released delivery cue.

    This tactical task supplies the intended line/length after release, before
    contact. It is not a frame-by-frame reaction task. Noise and outcomes remain
    hidden. Opponent bowling is chosen BEFORE the batting action; no hindsight.
    """
    metadata = {'render_modes': ['ansi']}

    def __init__(self, bowling_policy='random', **config):
        self.base = CricketEnv(**config)
        self.bowling_policy = bowling_policy
        self.action_space = spaces.Discrete(45)
        self.observation_space = spaces.Box(0, 1, shape=(BATTING_OBSERVATION_SIZE,), dtype=np.float32)
        self.pending_delivery = None
        self.render_mode = self.base.render_mode

    def _queue(self):
        from cricket_rl.experiment import baseline
        if isinstance(self.bowling_policy, str):
            self.pending_delivery = baseline(self.base, self.bowling_policy, self.opponent_rng)
        else:
            self.pending_delivery = int(self.bowling_policy.predict(self.base._observation(), deterministic=True)[0])
        if not self.base.action_space.contains(self.pending_delivery):
            raise ValueError('Bowling policy returned an invalid delivery')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Independent policy RNG: opponent randomness never depends on bat action.
        streams = np.random.SeedSequence(int(self.np_random.integers(2**32))).spawn(1)
        self.opponent_rng = np.random.default_rng(streams[0])
        _, info = self.base.reset(seed=seed, options=options)
        self._queue()
        return batting_observation(self.base, self.pending_delivery), info | {'role': 'batting'}

    def step(self, action):
        shot = Shot.decode(action)
        if self.base.done or self.pending_delivery is None:
            raise RuntimeError('Reset before stepping a finished innings')
        bowling_action = self.pending_delivery
        _, reward, terminated, truncated, info = self.base.execute(bowling_action, shot=shot)
        self.pending_delivery = None
        if not (terminated or truncated):
            self._queue()
        return batting_observation(self.base, self.pending_delivery), -reward, terminated, truncated, info | {
            'role': 'batting', 'bowling_action': bowling_action, 'batting_action': int(action)}

    def render(self):
        return self.base.render()

    def close(self):
        self.base.close()
