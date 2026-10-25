"""Train PPO and evaluate seeded policies with full experiment provenance."""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from cricket_rl.env import CricketEnv, OBSERVATION_VERSION, OBSERVATION_SIZE


EVALUATION_SEED = 100_000


def baseline(env, name, rng):
    if name == "random":
        return int(rng.integers(9))
    if name.startswith("fixed:"):
        action = int(name.split(":")[1])
        if not env.action_space.contains(action):
            raise ValueError("Invalid fixed action")
        return action
    if name == "varying":
        plan = env.batting_plan
        # Contrast the two stump-threatening lengths when one is anticipated.
        return 4 if plan.expected_action == 3 and plan.confidence > 0.25 else 3
    if name != "heuristic":
        raise ValueError("Unknown baseline")
    scores = env.field.copy()
    scores[env.preference] -= 1
    return int(np.argmax(scores)) * 3 + 1


def evaluate(
    policy, episodes=100, seed=EVALUATION_SEED, styles=("unseen",), **env_config
):
    if episodes < 2:
        raise ValueError("At least two evaluation episodes required")
    env = CricketEnv(styles=styles, **env_config)
    rows = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        rng = np.random.default_rng(seed + episode)
        done, reward = False, 0.0
        actions = [0] * 9
        while not done:
            action = (
                baseline(env, policy, rng)
                if isinstance(policy, str)
                else int(policy.predict(obs, deterministic=True)[0])
            )
            actions[action] += 1
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward += r
        rows.append(
            {
                **info,
                "reward": reward,
                "seed": seed + episode,
                "truncated": truncated,
                "actions": actions,
                "pitch": asdict(env.pitch),
            }
        )
    env.close()
    return rows


def summarize(rows):
    if len(rows) < 2:
        raise ValueError("At least two episodes required for uncertainty estimates")
    result = {}
    for key in ("runs", "wickets", "balls", "reward", "extras"):
        values = np.array([r[key] for r in rows], dtype=float)
        result[key] = {
            "mean": float(values.mean()),
            "ci95_half_width": float(1.96 * values.std(ddof=1) / np.sqrt(len(values))),
        }
    balls = sum(r["balls"] for r in rows)
    result["runs_per_over"] = 6 * sum(r["runs"] for r in rows) / max(1, balls)
    result["wickets_per_over"] = 6 * sum(r["wickets"] for r in rows) / max(1, balls)
    if all("actions" in r for r in rows):
        counts = np.sum([r["actions"] for r in rows], axis=0)
        result["action_counts"] = counts.tolist()
        result["dominant_action_fraction"] = float(counts.max() / max(1, counts.sum()))
    result["truncated_episodes"] = sum(r["truncated"] for r in rows)
    return result


def provenance():
    root = Path(__file__).parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    return {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "gymnasium")},
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()),
        "observation_version": OBSERVATION_VERSION,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("*.py"))
        },
    }


def load_model(path):
    """Reject older observation layouts before making an invalid prediction."""
    from stable_baselines3 import PPO

    model = PPO.load(path, device="cpu")
    if (
        model.observation_space.shape != (OBSERVATION_SIZE,)
        or getattr(model.action_space, "n", None) != 9
    ):
        raise ValueError(
            f"Checkpoint is incompatible with observation version {OBSERVATION_VERSION} "
            f"({OBSERVATION_SIZE} values); retrain with this simulator"
        )
    return model


def train(steps, seed, output, single_style=False, randomize_pitch=False, **config):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    import torch

    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = Monitor(
        CricketEnv(
            styles=("balanced",)
            if single_style
            else ("defensive", "balanced", "aggressive"),
            randomize_pitch=randomize_pitch,
            **config,
        ),
        str(output / "training"),
    )
    model = PPO(
        "MlpPolicy", env, seed=seed, n_steps=256, batch_size=64, device="cpu", verbose=0
    )
    start = time.monotonic()
    model.learn(total_timesteps=steps)
    model.save(output / "ppo")
    metadata = {
        "seed": seed,
        "requested_steps": steps,
        "actual_steps": model.num_timesteps,
        "single_style": single_style,
        "randomize_pitch": randomize_pitch,
        "env_config": config,
        "seconds": time.monotonic() - start,
        "provenance": provenance(),
        "packages": {name: version(name) for name in ("stable-baselines3", "torch")},
        "ppo": {"n_steps": 256, "batch_size": 64, "device": "cpu"},
    }
    (output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("runs/baselines"))
    parser.add_argument("--load", type=Path)
    parser.add_argument("--single-style", action="store_true")
    parser.add_argument("--static-batting", action="store_true")
    parser.add_argument("--randomize-pitch", action="store_true")
    parser.add_argument(
        "--pitch", choices=("standard", "green", "dry"), default="standard"
    )
    parser.add_argument("--wicket-reward", type=float, default=8.0)
    args = parser.parse_args()
    if args.episodes < 2 or args.train_steps < 0 or args.seed < 0 or args.eval_seed < 0:
        parser.error("Need >=2 episodes, nonnegative steps and seeds")
    if args.load and args.train_steps:
        parser.error("Choose training or loading a checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    policies = {"random": "random", "heuristic": "heuristic"}
    if args.train_steps:
        policies["ppo"] = train(
            args.train_steps,
            args.seed,
            args.output,
            args.single_style,
            args.randomize_pitch,
            wicket_reward=args.wicket_reward,
            adaptive_batting=not args.static_batting,
        )
    elif args.load:
        policies["ppo"] = load_model(args.load)
    report = {
        "provenance": provenance(),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "protocol": "Held-out lineup; matched initial-state seeds; normal approximate episode-mean CIs",
        "results": {},
    }
    for name, policy in policies.items():
        rows = evaluate(
            policy,
            args.episodes,
            args.eval_seed,
            pitch=args.pitch,
            wicket_reward=args.wicket_reward,
            adaptive_batting=not args.static_batting,
        )
        report["results"][name] = {"summary": summarize(rows), "episodes": rows}
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {name: value["summary"] for name, value in report["results"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
