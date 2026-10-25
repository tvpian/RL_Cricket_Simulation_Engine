"""Agent-controlled batting: stance × shot direction × power."""

from dataclasses import dataclass
import numpy as np
from cricket_rl.validation import integer

DIRECTIONS = (-1.2, -0.6, 0.0, 0.6, 1.2)
POWERS = (0.25, 0.55, 0.85)


@dataclass(frozen=True)
class Shot:
    stance: int
    direction: int
    power: int

    def __post_init__(self):
        integer(self.stance, "Stance", 0, 2)
        integer(self.direction, "Direction", 0, 4)
        integer(self.power, "Power", 0, 2)

    @classmethod
    def decode(cls, action):
        integer(action, "Batting action", 0, 44)
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 45:
            raise ValueError("Batting action must be an integer in [0, 44]")
        stance, rest = divmod(int(action), 15)
        direction, power = divmod(rest, 3)
        return cls(stance, direction, power)

    def encode(self):
        return self.stance * 15 + self.direction * 3 + self.power
