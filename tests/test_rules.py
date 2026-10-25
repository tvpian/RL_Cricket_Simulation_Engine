import numpy as np
import pytest
from cricket_rl.rules import Innings
from cricket_rl.physics import Delivery, Pitch, pursue, simulate
from cricket_rl import CricketEnv


def test_extras_free_hit_and_legal_ball_accounting():
    innings = Innings()
    innings.apply(Delivery(extras=1, extra_type='no_ball', legal=False))
    assert innings.free_hit and innings.balls == 0 and innings.runs == 1
    innings.apply(Delivery(extras=1, extra_type='wide', legal=False))
    assert innings.free_hit and innings.balls == 0
    with pytest.raises(ValueError, match='Illegal dismissal'):
        innings.apply(Delivery(wicket=True, dismissal='caught'))
    innings.apply(Delivery(runs=2))
    assert not innings.free_hit and innings.balls == 1
    assert innings.runs == 4 and innings.extras == 2 and innings.attempts == 3


def test_strike_rotation_and_new_batter():
    innings = Innings()
    innings.apply(Delivery(runs=1))
    assert (innings.striker, innings.non_striker) == (1, 0)
    for _ in range(5):
        innings.apply(Delivery())
    assert (innings.striker, innings.non_striker) == (0, 1)
    innings.apply(Delivery(wicket=True, dismissal='bowled'))
    assert innings.striker == 2 and innings.wickets == 1


def test_run_out_on_no_ball_and_target_completion():
    innings = Innings(target=4)
    innings.apply(Delivery(runs=1, extras=1, extra_type='no_ball', legal=False,
                          wicket=True, dismissal='run_out'))
    assert innings.wickets == 1 and innings.runs == 2
    innings.apply(Delivery(runs=2))
    assert innings.complete and innings.balls == 1
    with pytest.raises(RuntimeError):
        innings.apply(Delivery())


def test_ten_wickets_do_not_create_eleventh_striker():
    innings = Innings(wicket_limit=10)
    for _ in range(10):
        innings.apply(Delivery(wicket=True, dismissal='bowled'))
    assert innings.complete and innings.striker <= 10


def test_pursuit_speed_and_no_overshoot():
    points = np.zeros((9, 2))
    moved = pursue(points, np.array([10., 0.]), dt=.1, speed=6)
    np.testing.assert_allclose(moved[:, 0], .6)
    moved = pursue(points, np.array([.1, 0.]), dt=.1, speed=6)
    np.testing.assert_allclose(moved[:, 0], .1)


def test_pitch_validation_and_field_constraints():
    for config in ({'boundary': 10}, {'restitution': 1.2}, {'pace': float('nan')}):
        with pytest.raises(ValueError):
            Pitch(**config)
    env = CricketEnv(pitch='dry', record=True, target=50)
    obs, _ = env.reset(seed=12)
    assert env.observation_space.contains(obs)
    with pytest.raises(ValueError):
        env.set_field([[90, 0]] * 9)
    with pytest.raises(ValueError):
        env.set_field([[0, 0]])


def test_recording_does_not_change_outcomes_and_pursuit_is_bounded():
    env = CricketEnv()
    env.reset(seed=1)
    args = (0, 1, .8, .6, 1, env.fielders)
    a = simulate(np.random.default_rng(20), *args, record=True)
    b = simulate(np.random.default_rng(20), *args, record=False)
    assert (a.runs, a.wicket, a.dismissal, a.extras) == (b.runs, b.wicket, b.dismissal, b.extras)
    trace = np.array(a.trajectory)
    field = np.array(a.fielder_trajectory)
    assert len(trace) == len(field) and np.all(trace[:, 2] >= 0)
    if len(field) > 1:
        assert np.max(np.linalg.norm(np.diff(field, axis=0), axis=2)) <= .120001


def test_protected_deliveries_cannot_have_bowled_lbw_or_caught():
    env = CricketEnv()
    env.reset(seed=1)
    for seed in range(50):
        d = simulate(np.random.default_rng(seed), 1, 0, .8, .7, 1,
                     env.fielders, free_hit=True, record=False)
        assert d.dismissal not in ('bowled', 'lbw', 'caught')
