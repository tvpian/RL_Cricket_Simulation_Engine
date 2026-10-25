"""Agent-controlled batting: stance × shot direction × power."""
from dataclasses import dataclass
import numpy as np

DIRECTIONS = (-1.2, -.6, 0., .6, 1.2)
POWERS = (.25, .55, .85)


@dataclass(frozen=True)
class Shot:
    stance: int
    direction: int
    power: int

    @classmethod
    def decode(cls, action):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 45:
            raise ValueError('Batting action must be an integer in [0, 44]')
        stance, rest = divmod(int(action), 15)
        direction, power = divmod(rest, 3)
        return cls(stance, direction, power)

    def encode(self):
        return self.stance * 15 + self.direction * 3 + self.power
