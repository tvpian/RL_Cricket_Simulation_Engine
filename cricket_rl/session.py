"""Shared playground session used by HTTP and the browser Python runtime."""
from dataclasses import asdict
import numpy as np
from cricket_rl import CricketEnv, BattingEnv
from cricket_rl.api import EnvironmentAPI
from cricket_rl.roles import batting_baseline
from cricket_rl.experiment import baseline

class Session:
    def __init__(self, model=None, batting_model=None):
        self.batting_model = batting_model
        self.environments = EnvironmentAPI()
        self.model = model
        self.reset({})

    def reset(self, config):
        seed = config.get('seed', 42)
        overs = config.get('overs', 4)
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError('Seed must be an integer in [0, 2^32)')
        if not isinstance(overs, int) or not 1 <= overs <= 50:
            raise ValueError('Overs must be an integer in [1, 50]')
        role = config.get('role', 'bowling')
        if role not in ('bowling', 'batting'):
            raise ValueError('Unknown role')
        factory = BattingEnv if role == 'batting' else CricketEnv
        env = factory(styles=(config.get('style', 'balanced'),), balls=overs * 6,
                         max_wickets=10, pitch=config.get('pitch', 'standard'),
                         target=config.get('target'), record=True,
                         adaptive_batting=config.get('adaptive_batting', True))
        env.reset(seed=seed)
        self.settings = dict(role=role,seed=seed,overs=overs,style=config.get('style','balanced'),
                             pitch=config.get('pitch','standard'),target=config.get('target'),
                             adaptive_batting=config.get('adaptive_batting',True))
        self.role = role
        self.batting_env = env if role == 'batting' else None
        self.env = env.base if role == 'batting' else env
        self.seed, self.history = seed, []
        self.rng = np.random.default_rng(seed)
        return self.state()

    def state(self):
        e = self.env
        return dict(role=self.role, settings=self.settings, ball_limit=e.balls, wicket_value=e.wicket_reward, observation=(self.batting_env.observation_space.shape[0] if self.batting_env else 57),
                    pending_delivery=self.batting_env.pending_delivery if self.batting_env else None,
                    score=e._info(), seed=self.seed, done=e.done,
                    fielders=e.fielders.tolist(), pitch=asdict(e.pitch),
                    batter=dict(style=e.style, skill=e.skill, aggression=e.aggression,
                                handedness=e.handedness, preference=e.preference),
                    history=self.history, model_available=(self.batting_model is not None if self.role == 'batting' else self.model is not None),
                    target=e.target)

    def bowl(self, config):
        if self.role == 'batting':
            return self.bat(config)
        mode = config.get('policy', 'manual')
        if mode == 'manual':
            action = config.get('action', 4)
        elif mode in ('random', 'heuristic'):
            action = baseline(self.env, mode, self.rng)
        elif mode == 'ppo' and self.model is not None:
            action = int(self.model.predict(self.env._observation(), deterministic=True)[0])
        else:
            raise ValueError('Unknown or unavailable policy')
        if self.env.done:
            raise ValueError('Innings complete; start a new innings')
        _, reward, _, _, info = self.env.step(action)
        d = self.env.last_delivery
        self.history.append(dict(action=int(action), reward=reward, policy=mode, **info))
        # Return full traces; the browser performs playback, no physics duplication.
        return self.state() | {'delivery': asdict(d), 'reward': reward, 'action': int(action)}

    def bat(self, config):
        if self.env.done:
            raise ValueError('Innings complete; start a new innings')
        mode = config.get('policy', 'manual')
        if mode == 'manual':
            action = config.get('action', 22)
        elif mode in ('random', 'heuristic'):
            action = batting_baseline(self.batting_env, 'random' if mode == 'random' else 'reactive', self.rng)
        elif mode == 'ppo' and self.batting_model is not None:
            from cricket_rl.roles import batting_observation
            action = int(self.batting_model.predict(batting_observation(self.env, self.batting_env.pending_delivery), deterministic=True)[0])
        else:
            raise ValueError('Unknown or unavailable batting policy')
        _, reward, _, _, info = self.batting_env.step(action)
        self.history.append(dict(action=int(action), reward=reward, policy=mode, **info))
        return self.state() | {'delivery': asdict(self.env.last_delivery), 'reward': reward, 'action': int(action)}
