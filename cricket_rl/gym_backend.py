"""Select real Gymnasium on Python; a narrow protocol adapter in WebAssembly."""

import sys

if sys.platform == "emscripten":
    from types import SimpleNamespace
    from cricket_rl.browser_protocol import Env, Box, Discrete

    gym = SimpleNamespace(Env=Env)
    spaces = SimpleNamespace(Box=Box, Discrete=Discrete)
else:
    import gymnasium as gym
    from gymnasium import spaces

__all__ = ["gym", "spaces"]
