"""Isolated Gym reset/step sessions for the local development HTTP API."""
from uuid import uuid4
from cricket_rl.course_experiment import make_env


class EnvironmentAPI:
    def __init__(self):
        self.sessions = {}

    @staticmethod
    def schema(role):
        if role not in ('batting', 'bowling'):
            raise ValueError('Unknown role')
        return {'role': role, 'observation_size': 66 if role == 'batting' else 57,
                'actions': 45 if role == 'batting' else 9,
                'action_encoding': 'stance*15 + direction*3 + power' if role == 'batting' else 'line*3 + length',
                'observation_version': 1 if role == 'batting' else 3}

    def create(self, data):
        if len(self.sessions) >= 32:
            raise ValueError('Close an existing session before creating more')
        role, config = data.get('role', 'bowling'), data.get('config', {})
        self.schema(role)
        if not isinstance(config, dict):
            raise ValueError('config must be an object')
        # Configuration is explicit; API cannot load arbitrary local checkpoints.
        allowed = {'balls', 'max_wickets', 'pitch', 'target', 'adaptive_batting', 'wicket_reward'}
        if set(config) - allowed:
            raise ValueError('Unsupported environment configuration')
        env = make_env(role, 'random' if role == 'batting' else 'scripted', **config)
        obs, info = env.reset(seed=data.get('seed'))
        identifier = uuid4().hex
        self.sessions[identifier] = env
        return self.schema(role) | {'id': identifier, 'observation': obs.tolist(), 'info': info}

    def command(self, identifier, command, data):
        if identifier not in self.sessions:
            raise ValueError('Unknown or closed environment session')
        env = self.sessions[identifier]
        if command == 'reset':
            if data.get('options') is not None and not isinstance(data['options'], dict):
                raise ValueError('options must be an object')
            obs, info = env.reset(seed=data.get('seed'), options=data.get('options'))
            return {'observation': obs.tolist(), 'info': info}
        if command == 'step':
            obs, reward, terminated, truncated, info = env.step(data.get('action'))
            return dict(observation=obs.tolist(), reward=reward, terminated=terminated,
                        truncated=truncated, info=info)
        if command == 'close':
            env.close()
            del self.sessions[identifier]
            return {'closed': True}
        raise ValueError('Unknown environment operation')
