import numpy as np
import pytest
from cricket_rl import CricketEnv
from cricket_rl.batting import BattingPlan, BowlingMemory
from cricket_rl.physics import simulate


def test_memory_learns_then_forgets_without_future_information():
    memory = BowlingMemory()
    assert memory.plan() == BattingPlan()
    for _ in range(6):
        memory.observe(3)
    repeated = memory.plan()
    assert repeated.expected_action == 3 and repeated.confidence > .7
    ready, surprise = repeated.response(1, 0)
    assert ready == repeated.confidence and surprise == 0
    wrong_ready, wrong_surprise = repeated.response(2, 2)
    assert wrong_ready < ready and wrong_surprise > .6
    for _ in range(12):
        memory.observe(8)
    assert memory.plan().expected_action == 8
    assert memory.observation().shape == (19,)
    with pytest.raises(ValueError):
        memory.observe(-1)


def test_memory_is_observed_per_batter_and_reset():
    env = CricketEnv()
    obs, _ = env.reset(seed=12)
    assert len(obs) == 57
    env.memories[0].observe(3)
    changed = env._observation()
    assert not np.array_equal(obs[37:], changed[37:])
    env.innings.striker = 1
    assert env.batting_plan.expected_action is None
    env.reset(seed=12)
    np.testing.assert_array_equal(obs, env._observation())


def test_delivery_uses_history_before_current_action(monkeypatch):
    from cricket_rl.physics import Delivery
    seen = []
    def fake(*args, **kwargs):
        seen.append(kwargs['batting_plan'])
        return Delivery()
    monkeypatch.setattr('cricket_rl.env.simulate', fake)
    env = CricketEnv()
    env.reset(seed=9)
    env.step(3)
    assert seen[0] == BattingPlan()
    env.step(8)
    assert seen[1].expected_action == 3
    assert env.memories[0].last_action == 8


def test_anticipation_changes_contact_and_surprise_with_fixed_seeds():
    env = CricketEnv()
    env.reset(seed=1)
    counts = []
    for plan in (BattingPlan(), BattingPlan(3, .85), BattingPlan(8, .85)):
        contacts = 0
        for seed in range(100):
            result = simulate(np.random.default_rng(seed), 1, 0, .55, .6, 0,
                              env.fielders, batting_plan=plan)
            contacts += result.contact_frame is not None
        counts.append(contacts)
    assert counts[1] > counts[0] > counts[2]


def test_static_ablation_ignores_history_but_keeps_it_observable():
    env = CricketEnv(adaptive_batting=False)
    env.reset(seed=1)
    for _ in range(6):
        env.memories[0].observe(3)
    assert env.batting_plan == BattingPlan()
    assert env._observation()[-1] == 0
    assert env.memories[0].observation().sum() > 0


def test_static_control_preserves_original_seeded_rollouts():
    from cricket_rl.experiment import evaluate
    rows = evaluate('fixed:4', episodes=2, adaptive_batting=False)
    # Archived v2 rollouts: static physics must not drift when adding memory.
    expected = [(0, 3, 5, 0, 24.), (84, 1, 24, 0, -76.)]
    actual = [tuple(r[k] for k in ('runs', 'wickets', 'balls', 'extras', 'reward'))
              for r in rows]
    assert actual == expected


def test_history_probe_distinguishes_fixed_and_history_aware_policies():
    from cricket_rl.diagnostics import probe
    assert probe('fixed:3', states=3)['history_changed_action_fraction'] == 0
    assert probe('varying', states=3)['history_changed_action_fraction'] == 1
