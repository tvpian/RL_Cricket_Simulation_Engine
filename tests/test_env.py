import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from cricket_rl import CricketEnv
from cricket_rl.experiment import evaluate, summarize


def test_gym_contract():
    check_env(CricketEnv(), skip_render_check=True)


def test_seeded_rollout_and_accounting():
    a, b = CricketEnv(), CricketEnv()
    np.testing.assert_array_equal(a.reset(seed=42)[0], b.reset(seed=42)[0])
    total_reward = 0
    while True:
        x, y = a.step(4), b.step(4)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:] == y[1:]
        total_reward += x[1]
        assert a.observation_space.contains(x[0])
        if x[2]:
            break
    assert total_reward == 8 * a.wickets - a.runs
    assert a.deliveries == a.balls or a.wickets == a.max_wickets
    with pytest.raises(RuntimeError):
        a.step(0)


def test_probabilities_and_invalid_action():
    env = CricketEnv()
    env.reset(seed=3)
    with pytest.raises(ValueError):
        env.step(9)


def test_horizon_and_heldout_reproducibility():
    env = CricketEnv(balls=1)
    env.reset(seed=1)
    assert env.step(0)[2]
    rows = evaluate("heuristic", episodes=4)
    assert rows == evaluate("heuristic", episodes=4)
    assert all(r["style"] == "unseen" for r in rows)
    assert summarize(rows)["runs_per_over"] >= 0


def test_physics_trajectory_and_cricket_outcomes():
    from cricket_rl.physics import simulate

    rng = np.random.default_rng(123)
    fielders = np.array([[-20, 30], [0, 40], [20, 30]] * 3)
    for line in range(3):
        for length in range(3):
            result = simulate(rng, line, length, 0.8, 0.6, 1, fielders)
            trace = np.array(result.trajectory)
            assert trace.shape[1] == 3
            assert np.isfinite(trace).all()
            assert result.runs in (0, 1, 2, 3, 4, 6)
            assert not (result.runs and result.wicket and result.dismissal != "run_out")
            assert result.dismissal in (None, "bowled", "caught", "lbw", "run_out")
            if result.boundary:
                assert result.runs in (4, 6)
                assert np.linalg.norm(result.landing) >= 65


def test_configuration_and_reset_validation():
    for kwargs in (
        {"balls": 0},
        {"styles": ()},
        {"styles": ("unknown",)},
        {"render_mode": "rgb_array"},
    ):
        with pytest.raises(ValueError):
            CricketEnv(**kwargs)
    env = CricketEnv()
    with pytest.raises(ValueError):
        env.reset(options={"style": "unknown"})


def test_replay_export(tmp_path, monkeypatch):
    import sys
    from cricket_rl.replay import main

    output = tmp_path / "replay.html"
    monkeypatch.setattr(sys, "argv", ["replay", "--output", str(output)])
    main()
    html = output.read_text()
    assert "const data={" in html
    assert "requestAnimationFrame(draw)" in html
