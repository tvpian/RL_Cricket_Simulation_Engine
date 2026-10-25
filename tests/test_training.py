"""Meaningful PPO/checkpoint round-trip; skipped without optional training deps."""
import pytest
pytest.importorskip('stable_baselines3')
from stable_baselines3 import PPO
from cricket_rl.experiment import evaluate, train


def test_train_save_reload_reproduces_evaluation(tmp_path):
    model = train(256, seed=9, output=tmp_path)
    before = evaluate(model, episodes=2)
    restored = PPO.load(tmp_path / 'ppo.zip', device='cpu')
    assert evaluate(restored, episodes=2) == before
    assert model.num_timesteps == 256
    assert (tmp_path / 'training.monitor.csv').exists()
    assert (tmp_path / 'training.json').exists()
    from cricket_rl.server import Session
    session = Session(restored)
    assert session.bowl({'policy': 'ppo'})['score']['attempts'] == 1


def test_old_checkpoint_is_rejected_with_clear_upgrade_message(tmp_path):
    import gymnasium as gym
    import numpy as np
    from cricket_rl.experiment import load_model
    class OldEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 1, shape=(37,), dtype=np.float32)
        action_space = gym.spaces.Discrete(9)
    old = PPO('MlpPolicy', OldEnv(), n_steps=256, device='cpu')
    old.save(tmp_path / 'old')
    with pytest.raises(ValueError, match='observation version 3'):
        load_model(tmp_path / 'old.zip')


@pytest.mark.parametrize('algorithm',['ppo','dqn'])
def test_trained_batsman_and_frozen_neural_opponents(algorithm,tmp_path):
    from cricket_rl.course_experiment import train as role_train,load_policy
    from cricket_rl import BattingEnv,CricketEnv
    bat=role_train('batting',algorithm,256,seed=5,output=tmp_path)
    restored=load_policy(tmp_path/'policy.zip','batting')
    env=BattingEnv()
    obs,_=env.reset(seed=4)
    assert bat.predict(obs,deterministic=True)[0]==restored.predict(obs,deterministic=True)[0]
    bowl=CricketEnv(batting_policy=restored)
    bowl.reset(seed=4)
    assert bowl.step(3)[4]['attempts']==1
