"""Run a fixed multi-seed generalization study, then generate an HTML report."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from cricket_rl.experiment import evaluate, provenance, summarize, train

SCENARIOS = {
    "in_distribution": {
        "styles": ("defensive", "balanced", "aggressive"),
        "pitch": "standard",
    },
    "heldout_batter": {"styles": ("unseen",), "pitch": "standard"},
    "green_pitch": {"styles": ("unseen",), "pitch": "green"},
    "static_control": {
        "styles": ("unseen",),
        "pitch": "standard",
        "adaptive_batting": False,
    },
    "dry_pitch": {"styles": ("unseen",), "pitch": "dry"},
}


def run_job(args):
    mode, seed, steps, episodes, root, reuse = args
    output = Path(root) / f"{mode}-{seed}"
    if reuse:
        from stable_baselines3 import PPO

        metadata = json.loads((output / "training.json").read_text())
        if (
            metadata["requested_steps"] != steps
            or metadata["seed"] != seed
            or metadata["single_style"] != (mode == "single")
            or metadata["randomize_pitch"]
        ):
            raise ValueError("Checkpoint configuration does not match requested study")
        current = provenance()["source_sha256"]
        saved = metadata["provenance"]["source_sha256"]
        if any(
            current[name] != saved[name]
            for name in ("env.py", "physics.py", "rules.py", "batting.py")
        ):
            raise ValueError(
                "Simulator changed; retrain rather than reusing checkpoints"
            )
        model = PPO.load(output / "ppo.zip", device="cpu")
    else:
        model = train(
            steps, seed, output, single_style=mode == "single", randomize_pitch=False
        )
    result = {}
    for name, scenario in SCENARIOS.items():
        rows = evaluate(model, episodes=episodes, **scenario)
        result[name] = {"summary": summarize(rows), "episodes": rows}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return mode, seed, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Re-evaluate matching retained checkpoints without retraining",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/adaptive-study"))
    args = parser.parse_args()
    if (
        args.steps < 256
        or args.episodes < 2
        or len(args.seeds) < 2
        or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error("Need >=256 steps, >=2 episodes, and >=2 distinct training seeds")
    if min(args.seeds) < 0:
        parser.error("Seeds must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "provenance": provenance(),
        "config": {"steps": args.steps, "seeds": args.seeds, "episodes": args.episodes},
        "baselines": {},
        "trained": {},
    }
    # Select a single constant action on a disjoint validation set, never test data.
    validation = {}
    for action in range(9):
        rows = evaluate(
            f"fixed:{action}",
            episodes=args.episodes,
            seed=50_000,
            styles=("defensive", "balanced", "aggressive"),
        )
        validation[action] = summarize(rows)["reward"]["mean"]
    selected = max(validation, key=validation.get)
    report["constant_selection"] = {
        "validation_seed": 50_000,
        "mean_rewards": validation,
        "selected_action": selected,
    }
    for scenario, config in SCENARIOS.items():
        report["baselines"][scenario] = {}
        for policy in ("random", "heuristic", "varying", f"fixed:{selected}"):
            rows = evaluate(policy, episodes=args.episodes, **config)
            report["baselines"][scenario][policy] = {
                "summary": summarize(rows),
                "episodes": rows,
            }
    print("Baselines evaluated on all five scenarios", flush=True)
    jobs = [
        (mode, seed, args.steps, args.episodes, str(args.output), args.reuse)
        for mode in ("single", "mixed")
        for seed in args.seeds
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(pending):
            mode, seed, results = future.result()
            report["trained"][f"{mode}-{seed}"] = results
            (args.output / "study.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"Completed {mode} seed {seed} ({len(report['trained'])}/{len(jobs)})",
                flush=True,
            )
    from cricket_rl.report import generate

    generate(args.output / "study.json", args.output / "report.html")
    print(f"Finished: {args.output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
