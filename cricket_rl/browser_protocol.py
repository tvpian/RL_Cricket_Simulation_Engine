"""Small browser-only Env/space adapter; not a replacement for desktop Gymnasium.

Pyodide cannot import Gymnasium's multiprocessing vector environments. The web
playground needs only seeded reset, scalar action validation and observation
bounds. Physics, rules and role environments remain shared with desktop Python.
Training and the registered Gym environments continue to use real Gymnasium.
"""
import numpy as np


class Env:
    _np_random = None

    @property
    def np_random(self):
        if self._np_random is None:
            self._np_random = np.random.default_rng()
        return self._np_random

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            if not isinstance(seed, (int, np.integer)) or seed < 0:
                raise ValueError('Seed must be a nonnegative integer')
            self._np_random = np.random.default_rng(seed)

    def close(self):
        pass


class Discrete:
    def __init__(self, n):
        self.n = n

    def contains(self, action):
        return isinstance(action, (int, np.integer)) and 0 <= action < self.n


class Box:
    def __init__(self, low, high, shape, dtype):
        self.low, self.high, self.shape, self.dtype = low, high, shape, np.dtype(dtype)

    def contains(self, observation):
        values = np.asarray(observation)
        return values.shape == self.shape and np.can_cast(values.dtype,self.dtype) and bool(np.all(values >= self.low) and np.all(values <= self.high))
