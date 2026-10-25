"""A transparent opponent model that prepares using past deliveries only.

The adaptation coefficients are hypotheses for experiments, not fitted batting
behavior. No bowler reward or repetition penalty is introduced here.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BattingPlan:
    expected_action: int | None = None
    confidence: float = 0.0

    def response(self, line, length):
        """Return readiness and surprise after the actual ball is revealed."""
        if self.expected_action is None:
            return 0.0, 0.0
        expected_line, expected_length = divmod(self.expected_action, 3)
        match = float(
            np.exp(
                -1.5 * (line - expected_line) ** 2
                - 0.8 * (length - expected_length) ** 2
            )
        )
        return self.confidence * match, self.confidence * (1 - match)


class BowlingMemory:
    """Per-batter exponentially decayed counts, reset with each innings.

    Nine weights and a ten-way last-action code (including no history) are the
    complete state. Unfaced batters start with no knowledge, even if teammates
    have seen the bowler. Weights retain 75% of previous evidence per delivery.
    """

    decay = 0.75
    prior = 0.02

    def __init__(self):
        self.weights = np.zeros(9, dtype=np.float64)
        self.last_action = 9

    def observe(self, action):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 9:
            raise ValueError("Bowling action must be an integer in [0, 8]")
        self.weights *= self.decay
        self.weights[int(action)] += 1 - self.decay
        self.last_action = int(action)

    def plan(self):
        if not self.weights.any():
            return BattingPlan()
        probabilities = (self.weights + self.prior) / (
            self.weights.sum() + 9 * self.prior
        )
        expected = int(np.argmax(probabilities))
        confidence = float(
            np.clip((probabilities[expected] - 1 / 9) / (1 - 1 / 9), 0, 1)
        )
        return BattingPlan(expected, confidence)

    def observation(self):
        last = np.zeros(10, dtype=np.float32)
        last[self.last_action] = 1
        return np.r_[self.weights, last].astype(np.float32)
