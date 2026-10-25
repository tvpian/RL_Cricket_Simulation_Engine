# Gymnasium and HTTP interfaces

Use in-process environments for training speed. HTTP is useful for remote
controllers, debugging and a web client. Both routes execute the same physics;
there is no separate browser outcome engine. The server is single-user development
software, bound to loopback by default, with at most 32 isolated Gym sessions.

## Registered environments

```python
import gymnasium as gym
import cricket_rl
bat = gym.make('CricketBatting-v0')
bowl = gym.make('CricketBowling-v0')
```

Batting action encoding: `stance*15 + direction*3 + power`. Stance indices are
leg/centre/off; directions are -1.2,-0.6,0,0.6,1.2 radians relative to handedness;
power is defensive/balanced/attacking. Timing error and player skill remain
stochastic; the agent controls tactical preparation, not muscle-level movement.
Bowling action encoding is `line*3 + length` with yorker/good/short lengths.

## HTTP operations

Start `python -m cricket_rl.server` and use these JSON endpoints:

| Method/path | Body | Result |
|---|---|---|
| GET `/api/schema` | — | Both role schemas |
| POST `/api/envs` | `{"role":"batting","seed":42,"config":{"balls":6}}` | Session `id`, observation, info and spaces |
| POST `/api/envs/{id}/reset` | `{"seed":42}` | Observation and info |
| POST `/api/envs/{id}/step` | `{"action":22}` | Observation, reward, terminated, truncated, info |
| POST `/api/envs/{id}/close` | `{}` | Closes that session |

Separate Gym sessions cannot advance or reset one another. They are also separate
from the dashboard's match. Close sessions after use. JSON config permits ball and
wicket limits, pitch, target, adaptation flag and wicket reward. It cannot load
arbitrary local model files; checkpoint loading is a server-startup operation.

A Gym adapter is included:

```python
from cricket_rl.remote import RemoteEnv

env = RemoteEnv('http://127.0.0.1:8000', role='batting', balls=6)
try:
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(22)
finally:
    env.close()
```

That loopback address is an example for a locally running server. Cloud users need
their environment's supported port-forwarding mechanism; no public hosting or
production authentication is supplied here.

The dashboard retains `/api/reset`, `/api/bowl`, `/api/state`, and `/api/field`.
For compatibility, `/api/bowl` executes whichever role was selected at reset. The
body `policy: "ppo"` selects the loaded policy, including Q-learning or DQN; new
controllers should use the explicit Gym session endpoints above.
