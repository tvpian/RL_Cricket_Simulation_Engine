"""Regression cases from the public-release audit, not performance guarantees."""

import numpy as np
import pytest
from cricket_rl import CricketEnv, BattingEnv
from cricket_rl.api import EnvironmentAPI
from cricket_rl.course_experiment import QLearning, train, load_policy
from cricket_rl.physics import Pitch, Delivery, simulate
from cricket_rl.rules import Innings


def test_q_bootstraps_truncation_but_not_natural_termination():
    env = BattingEnv()
    obs, _ = env.reset(seed=42)
    model = QLearning("batting")
    model.q[:] = 10
    model.update(obs, 0, 1, obs, terminated=False, learning_rate=1, gamma=0.9)
    assert model.q[model.state(obs), 0] == 10
    model.update(obs, 0, 1, obs, terminated=True, learning_rate=1, gamma=0.9)
    assert model.q[model.state(obs), 0] == 1


def test_truncated_batting_observation_retains_a_real_delivery_cue(monkeypatch):
    env = BattingEnv(balls=1)
    env.reset(seed=4)
    env.base.innings.attempts = 3
    monkeypatch.setattr(
        "cricket_rl.env.simulate",
        lambda *a, **k: Delivery(extras=1, extra_type="wide", legal=False),
    )
    obs, _, terminated, truncated, _ = env.step(22)
    assert truncated and not terminated
    assert obs[-9:].sum() == 1
    with pytest.raises(RuntimeError):
        env.step(22)


def test_invalid_api_reset_does_not_change_existing_episode():
    api = EnvironmentAPI()
    identifier = api.create({"role": "batting", "seed": 42})["id"]
    original = api.sessions[identifier]
    before = original.base.np_random.bit_generator.state
    with pytest.raises(ValueError):
        api.command(
            identifier, "reset", {"seed": 0, "options": {"fielders": [[100, 100]] * 9}}
        )
    assert api.sessions[identifier] is original
    assert original.base.np_random.bit_generator.state == before
    assert original.base.innings.attempts == 0


def test_extreme_friction_has_bounded_integration():
    base = CricketEnv()
    base.reset(seed=42)
    with pytest.raises(RuntimeError, match="six seconds"):
        simulate(
            np.random.default_rng(1),
            1,
            2,
            0.5,
            0.5,
            1,
            base.fielders,
            Pitch(friction=0.999),
            record=False,
        )


@pytest.mark.parametrize("action", [True, 1.5, float("nan"), -1, 9])
def test_bowling_rejects_invalid_actions_before_mutating_state(action):
    env = CricketEnv()
    env.reset(seed=4)
    before = env.np_random.bit_generator.state
    with pytest.raises(ValueError):
        env.step(action)
    assert env.innings.attempts == 0
    assert env.np_random.bit_generator.state == before


def test_scoring_rejects_inconsistent_delivery_before_mutation():
    innings = Innings()
    with pytest.raises(ValueError):
        innings.apply(Delivery(runs=2.5))
    with pytest.raises(ValueError):
        innings.apply(Delivery(extras=1, extra_type="wide", legal=True))
    assert innings.runs == 0 and innings.attempts == 0


def test_run_outputs_cannot_silently_overwrite_a_checkpoint(tmp_path):
    train("batting", "qlearning", 32, 4, tmp_path)
    original = (tmp_path / "policy.npz").read_bytes()
    with pytest.raises(ValueError, match="already exist"):
        train("batting", "qlearning", 32, 4, tmp_path)
    assert (tmp_path / "policy.npz").read_bytes() == original


def test_nonfinite_q_checkpoint_is_rejected(tmp_path):
    np.savez(tmp_path / "bad.npz", q=np.full((45, 45), np.nan), role="batting")
    with pytest.raises(ValueError, match="Invalid Q table"):
        load_policy(tmp_path / "bad.npz", "batting")
