import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import cricket_rl
from cricket_rl import BattingEnv, CricketEnv
from cricket_rl.shots import Shot
from cricket_rl.course_experiment import QLearning, evaluate, train, load_policy


def test_registered_roles_and_batting_contract():
    check_env(BattingEnv(), skip_render_check=True)
    for name, size, actions in [('CricketBatting-v0',66,45),('CricketBowling-v0',57,9)]:
        env=gym.make(name)
        obs,_=env.reset(seed=12)
        assert obs.shape==(size,) and env.action_space.n==actions
        env.close()


def test_batting_is_seeded_and_delivery_is_committed_before_shot():
    a,b=BattingEnv(),BattingEnv()
    oa,_=a.reset(seed=10)
    ob,_=b.reset(seed=10)
    np.testing.assert_array_equal(oa,ob)
    cue=int(oa[-9:].argmax())
    x,y=a.step(22),b.step(22)
    assert x[4]['bowling_action']==cue
    np.testing.assert_array_equal(x[0],y[0])
    assert x[1:]==y[1:]
    assert a.observation_space.contains(x[0])
    before=a.base.innings.attempts
    with pytest.raises(ValueError):a.step(45)
    assert a.base.innings.attempts==before


def test_shared_physics_and_opposite_role_rewards():
    bat=BattingEnv(bowling_policy='fixed:3',record=True)
    bat.reset(seed=7)
    bowl=CricketEnv(record=True)
    bowl.reset(seed=7)
    x=bat.step(22)
    y=bowl.execute(3,shot=Shot.decode(22))
    assert x[1]==-y[1]
    assert bat.base.runs==bowl.runs and bat.base.wickets==bowl.wickets
    assert bat.base.last_delivery.trajectory==bowl.last_delivery.trajectory


def test_shots_change_contact_and_trajectories():
    from cricket_rl.physics import simulate
    env=CricketEnv()
    env.reset(seed=1)
    contacts=[]
    for stance in (0,1):
        contacts.append(sum(simulate(np.random.default_rng(seed),1,0,.55,.6,1,
                            env.fielders,shot=Shot(stance,2,1)).contact_frame is not None for seed in range(50)))
    assert contacts[1]>contacts[0]
    a=simulate(np.random.default_rng(5),1,1,.55,.6,1,env.fielders,shot=Shot(1,0,0))
    b=simulate(np.random.default_rng(5),1,1,.55,.6,1,env.fielders,shot=Shot(1,4,2))
    assert a.trajectory!=b.trajectory
    for i in range(45):assert Shot.decode(i).encode()==i


@pytest.mark.parametrize('role',['batting','bowling'])
def test_tabular_learning_checkpoint_roundtrip(role,tmp_path):
    model=train(role,'qlearning',256,7,tmp_path,'random' if role=='batting' else 'scripted')
    assert np.count_nonzero(model.q)>0
    restored=load_policy(tmp_path/'policy.npz',role)
    assert evaluate(role,model,episodes=2)==evaluate(role,restored,episodes=2)
    with pytest.raises(ValueError):load_policy(tmp_path/'policy.npz','bowling' if role=='batting' else 'batting')
