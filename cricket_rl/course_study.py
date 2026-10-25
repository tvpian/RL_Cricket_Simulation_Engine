"""Reproducible two-role Q-learning/PPO/DQN pilot with frozen-opponent matches."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import numpy as np
from cricket_rl.course_experiment import train, evaluate, load_policy
from cricket_rl.experiment import provenance
from cricket_rl.report import interval
from cricket_rl.artifacts import write_json


def job(args):
    role, algorithm, seed, steps, episodes, root, overwrite = args
    folder = Path(root) / f"{role}-{algorithm}-{seed}"
    model = train(
        role,
        algorithm,
        steps,
        seed,
        folder,
        "random" if role == "batting" else "scripted",
        overwrite=overwrite,
    )
    scenarios = {"familiar": {}, "heldout": {"pitch": "dry", "styles": ("unseen",)}}
    results = {}
    for name, config in scenarios.items():
        opponent = (
            ("random" if name == "familiar" else "varying")
            if role == "batting"
            else "scripted"
        )
        results[name] = evaluate(role, model, episodes, opponent=opponent, **config)
    (folder / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    return f"{role}-{algorithm}-{seed}", results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=8192)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output", type=Path, default=Path("runs/course-study"))
    args = p.parse_args()
    if (
        args.steps < 256
        or args.episodes < 2
        or len(set(args.seeds)) != len(args.seeds)
        or len(args.seeds) < 2
        or min(args.seeds) < 0
    ):
        p.error("Need >=256 steps, >=2 episodes, and >=2 distinct nonnegative seeds")
    if (args.output / "study.json").exists() and not args.overwrite:
        p.error("Study already exists; use a fresh output directory or --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "provenance": provenance(),
        "config": vars(args) | {"output": str(args.output)},
        "baselines": {},
        "trained": {},
    }
    for role in ("batting", "bowling"):
        report["baselines"][role] = {}
        for name in ("random", "reactive" if role == "batting" else "varying"):
            report["baselines"][role][name] = {
                scenario: evaluate(
                    role,
                    name,
                    args.episodes,
                    opponent=("random" if scenario == "familiar" else "varying")
                    if role == "batting"
                    else "scripted",
                    **(
                        {}
                        if scenario == "familiar"
                        else {"pitch": "dry", "styles": ("unseen",)}
                    ),
                )
                for scenario in ("familiar", "heldout")
            }
    jobs = [
        (role, algo, seed, args.steps, args.episodes, str(args.output), args.overwrite)
        for role in ("batting", "bowling")
        for algo in ("qlearning", "ppo", "dqn")
        for seed in args.seeds
    ]
    with ProcessPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(job, j) for j in jobs]):
            name, results = future.result()
            report["trained"][name] = results
            write_json(args.output / "study.json", report)
            print(f"{len(report['trained'])}/{len(jobs)}: {name}", flush=True)
    # Fixed seed and algorithm specified in advance; do not pick test winners.
    seed = args.seeds[0]
    paths = {
        r: args.output / f"{r}-ppo-{seed}" / "policy.zip"
        for r in ("batting", "bowling")
    }
    models = {r: load_policy(path, r) for r, path in paths.items()}
    report["frozen_opponents"] = {
        "artifacts": {
            r: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for r, path in paths.items()
        },
        "batting": evaluate(
            "batting", models["batting"], args.episodes, opponent=models["bowling"]
        ),
        "bowling": evaluate(
            "bowling", models["bowling"], args.episodes, opponent=models["batting"]
        ),
    }
    write_json(args.output / "study.json", report)
    lines = [
        "# Two-role course experiment",
        "",
        f"Budget: {len(jobs)} runs × {args.steps} requested transitions. {args.episodes} evaluation innings per scenario; seeds {args.seeds}. Test seeds begin at 300000. All seeds retained. This is a short pipeline pilot, not evidence of convergence.",
        "",
        "| Role | Algorithm | Scenario | Mean reward ± seed interval | Runs/innings | Wickets/innings |",
        "|---|---|---|---:|---:|---:|",
    ]
    summary = []
    for role in ("batting", "bowling"):
        for algo in ("qlearning", "ppo", "dqn"):
            for scenario in ("familiar", "heldout"):
                runs = [
                    report["trained"][f"{role}-{algo}-{seed}"][scenario]
                    for seed in args.seeds
                ]
                mean, ci = interval(
                    [np.mean([e["reward"] for e in run]) for run in runs]
                )
                row = dict(
                    role=role,
                    algorithm=algo,
                    scenario=scenario,
                    reward=mean,
                    ci95=ci,
                    runs=float(np.mean([e["runs"] for run in runs for e in run])),
                    wickets=float(np.mean([e["wickets"] for run in runs for e in run])),
                )
                summary.append(row)
                lines.append(
                    f"| {role} | {algo} | {scenario} | {mean:.2f} ± {ci:.2f} | {row['runs']:.2f} | {row['wickets']:.2f} |"
                )
    lines += [
        "",
        "Baselines (episode means; no training-seed interval):",
        "",
        "| Role | Baseline | Familiar reward | Held-out reward |",
        "|---|---|---:|---:|",
    ]
    for role, policies in report["baselines"].items():
        for name, scenarios in policies.items():
            lines.append(
                f"| {role} | {name} | {np.mean([r['reward'] for r in scenarios['familiar']]):.2f} | {np.mean([r['reward'] for r in scenarios['heldout']]):.2f} |"
            )
    lines += [
        "",
        "Reward is runs − 8 × wickets for batting, and its negative for bowling. Roles have different opponents and decision timing; their scores are not a ranking between roles. Held-out combines expert attributes, dry pitch, and (for batting) varying bowling. This is a bundled stress test, not an isolated causal ablation.",
        "",
        "Intervals are Student-t intervals across training-seed means, not a formal superiority test. The simulator remains uncalibrated. Frozen PPO seed-0 opponents were also exercised in both directions; see study.json. These matches are sequential fixed-policy evaluations, not simultaneous self-play or a win-rate league.",
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Complete", flush=True)


if __name__ == "__main__":
    main()
