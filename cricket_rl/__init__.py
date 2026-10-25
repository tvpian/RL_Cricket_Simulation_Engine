"""Two-role cricket strategy environments."""
import sys
from cricket_rl.env import CricketEnv
from cricket_rl.roles import BattingEnv

if sys.platform != 'emscripten':
    from gymnasium.envs.registration import register, registry
    for name, entry in [('CricketBowling-v0', 'cricket_rl.env:CricketEnv'),
                        ('CricketBatting-v0', 'cricket_rl.roles:BattingEnv')]:
        if name not in registry:
            register(id=name, entry_point=entry)

__all__ = ['CricketEnv', 'BattingEnv']
