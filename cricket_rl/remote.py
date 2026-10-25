"""Gymnasium adapter over the local HTTP API; use in-process envs for speed."""
import json
from urllib.request import Request, urlopen
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RemoteEnv(gym.Env):
    def __init__(self, base_url, role='batting', **config):
        self.base_url = base_url.rstrip('/')
        result = self._request('/api/envs', {'role': role, 'config': config})
        self.identifier = result['id']
        self.observation_space = spaces.Box(0, 1, shape=(result['observation_size'],), dtype=np.float32)
        self.action_space = spaces.Discrete(result['actions'])

    def _request(self, path, data):
        payload = json.dumps(data).encode()
        with urlopen(Request(self.base_url+path, data=payload, headers={'Content-Type': 'application/json'}), timeout=30) as response:
            return json.loads(response.read())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        result = self._request(f'/api/envs/{self.identifier}/reset', {'seed': seed, 'options': options})
        return np.array(result['observation'], dtype=np.float32), result['info']

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Action is outside the environment action space')
        result = self._request(f'/api/envs/{self.identifier}/step', {'action': int(action)})
        return (np.array(result['observation'], dtype=np.float32), result['reward'],
                result['terminated'], result['truncated'], result['info'])

    def close(self):
        if self.identifier:
            self._request(f'/api/envs/{self.identifier}/close', {})
            self.identifier = None
