"""Train either role, compare algorithms, and freeze opponents for fair evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from cricket_rl import CricketEnv, BattingEnv
from cricket_rl.experiment import baseline, provenance
from cricket_rl.roles import batting_baseline


def make_env(role, opponent=None, **config):
    opponent = opponent if opponent is not None else ('random' if role == 'batting' else 'scripted')
    if role == 'batting':
        return BattingEnv(bowling_policy=opponent, **config)
    if role == 'bowling':
        if isinstance(opponent, str) and opponent != 'scripted':
            raise ValueError('Bowling opponent must be scripted or a frozen batting model')
        return CricketEnv(batting_policy=None if isinstance(opponent, str) else opponent, **config)
    raise ValueError('Role must be batting or bowling')


def act(env, policy, obs, rng):
    if isinstance(policy, str):
        return batting_baseline(env, policy, rng) if isinstance(env, BattingEnv) else baseline(env, policy, rng)
    return int(policy.predict(obs, deterministic=True)[0])


class QLearning:
    """Small, inspectable discretization. Aliasing deliberately sacrifices detail."""
    def __init__(self, role, seed=0):
        self.role = role
        self.q = np.zeros((90 if role == 'bowling' else 45, 9 if role == 'bowling' else 45))
        self.rng = np.random.default_rng(seed)

    def state(self, obs):
        if self.role == 'bowling':
            weights = obs[37:46]
            expected = int(np.argmax(weights)) if weights.sum() > 0 else 9
            return expected * 9 + min(2, int(weights.max() * 3)) * 3 + int(round(obs[4] * 2))
        positions = (obs[8:26].reshape(9, 2) * 180) - 90
        angles = np.arctan2(positions[:, 0], positions[:, 1])
        candidates = np.array([-1.2, -.6, 0, .6, 1.2]) * (1 if obs[34] > .5 else -1)
        gap = int(np.argmin(np.exp(-((angles[:, None] - candidates) / .3)**2).sum(axis=0)))
        return int(np.argmax(obs[-9:])) * 5 + gap

    def predict(self, obs, deterministic=True):
        return np.int64(np.argmax(self.q[self.state(obs)])), None

    def save(self, path):
        np.savez(path, q=self.q, role=self.role)


def load_policy(path, role):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            if str(data['role']) != role:
                raise ValueError('Checkpoint role does not match')
            model = QLearning(role)
            if data['q'].shape != model.q.shape:
                raise ValueError('Invalid Q table')
            model.q[:] = data['q']
            return model
    from stable_baselines3 import PPO, DQN
    metadata_path = path.parent / 'training-role.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata['role'] != role:
        raise ValueError('Checkpoint role does not match')
    model = {'ppo': PPO, 'dqn': DQN}[metadata['algorithm']].load(path, device='cpu')
    expected = (66, 45) if role == 'batting' else (57, 9)
    if model.observation_space.shape != (expected[0],) or model.action_space.n != expected[1]:
        raise ValueError('Incompatible observation/action schema')
    return model


def train(role, algorithm, steps, seed, output, opponent=None):
    opponent = opponent if opponent is not None else ('random' if role == 'batting' else 'scripted')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_env(role, opponent)
    start = time.monotonic()
    if algorithm == 'qlearning':
        model = QLearning(role, seed)
        obs, _ = env.reset(seed=seed)
        episode, reward, length = 0, 0., 0
        logs = []
        for step in range(steps):
            epsilon = max(.05, 1 - step / max(steps * .8, 1))
            action = int(model.rng.integers(env.action_space.n)) if model.rng.random() < epsilon else int(model.predict(obs)[0])
            next_obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = model.state(obs)
            target = r + (0 if done else .99 * model.q[model.state(next_obs)].max())
            model.q[state, action] += .1 * (target - model.q[state, action])
            reward += r
            length += 1
            obs = next_obs
            if done:
                logs.append({'step': step+1, 'reward': reward, 'length': length})
                episode += 1
                obs, _ = env.reset(seed=seed + episode)
                reward, length = 0., 0
        model.save(output / 'policy.npz')
        (output / 'learning.json').write_text(json.dumps(logs))
        actual = steps
    else:
        import torch
        from stable_baselines3 import PPO, DQN
        from stable_baselines3.common.monitor import Monitor
        torch.set_num_threads(1)
        env = Monitor(env, str(output / 'training'))
        cls = {'ppo': PPO, 'dqn': DQN}[algorithm]
        params = dict(n_steps=256, batch_size=64) if algorithm == 'ppo' else dict(
            learning_starts=256, buffer_size=20_000, batch_size=64, train_freq=4, gradient_steps=1)
        model = cls('MlpPolicy', env, seed=seed, device='cpu', verbose=0, **params)
        model.learn(total_timesteps=steps)
        model.save(output / 'policy')
        actual = model.num_timesteps
    env.close()
    metadata = dict(role=role, algorithm=algorithm, seed=seed, requested_steps=steps,
                    actual_steps=actual, seconds=time.monotonic()-start,
                    opponent=opponent if isinstance(opponent, str) else 'frozen model', provenance=provenance())
    (output / 'training-role.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return model


def evaluate(role, policy, episodes=50, seed=300_000, opponent=None, **config):
    if episodes < 2:
        raise ValueError('Need at least two evaluation episodes')
    opponent = opponent if opponent is not None else ('varying' if role == 'batting' else 'scripted')
    env = make_env(role, opponent, **config)
    rows = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        rng = np.random.default_rng(seed + episode)
        reward, counts, done = 0., [0]*env.action_space.n, False
        while not done:
            action = act(env, policy, obs, rng)
            obs, r, terminated, truncated, info = env.step(action)
            counts[action] += 1
            reward += r
            done = terminated or truncated
        base = env.base if role == 'batting' else env
        rows.append(dict(seed=seed+episode, reward=reward, runs=base.runs,
                         wickets=base.wickets, balls=base.deliveries, actions=counts,
                         truncated=truncated, pitch=asdict(base.pitch)))
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['batting', 'bowling'], required=True)
    parser.add_argument('--algorithm', choices=['qlearning', 'ppo', 'dqn'], default='ppo')
    parser.add_argument('--steps', type=int, default=16384)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--opponent-model', type=Path)
    args = parser.parse_args()
    if args.steps < 256 or args.seed < 0 or args.episodes < 2:
        parser.error('Need >=256 steps, nonnegative seed and >=2 episodes')
    opponent = load_policy(args.opponent_model, 'bowling' if args.role == 'batting' else 'batting') if args.opponent_model else ('random' if args.role == 'batting' else 'scripted')
    model = train(args.role, args.algorithm, args.steps, args.seed, args.output, opponent)
    policies = {'random': 'random', 'baseline': 'reactive' if args.role == 'batting' else 'varying', args.algorithm: model}
    results = {name: evaluate(args.role, policy, args.episodes, opponent=opponent) for name, policy in policies.items()}
    (args.output / 'evaluation-role.json').write_text(json.dumps(results, indent=2)+'\n')
    print({name: float(np.mean([r['reward'] for r in rows])) for name, rows in results.items()})


if __name__ == '__main__':
    main()
