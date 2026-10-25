"""Matched-state probes isolate whether a policy responds to bowling history.

These are synthetic observation interventions, not additional performance games.
All match, batter and field features are held fixed while memory is replaced.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cricket_rl import CricketEnv
from cricket_rl.batting import BowlingMemory
from cricket_rl.experiment import baseline, load_model


def probe(policy, states=30):
    counts = {name: [0] * 9 for name in ('repeated_yorker', 'repeated_good_length')}
    changed = switched = 0
    env = CricketEnv(styles=('unseen',))
    for seed in range(200_000, 200_000 + states):
        env.reset(seed=seed)
        env.innings.balls = 12
        actions = []
        for name, previous in [('repeated_yorker', 3), ('repeated_good_length', 4)]:
            memory = BowlingMemory()
            for _ in range(6):
                memory.observe(previous)
            env.memories[env.innings.striker] = memory
            action = baseline(env, policy, np.random.default_rng(seed)) if isinstance(policy, str) else int(policy.predict(env._observation(), deterministic=True)[0])
            counts[name][action] += 1
            actions.append(action)
        changed += actions[0] != actions[1]
        switched += actions == [4, 3]
    env.close()
    return {'states': states, 'history_changed_action_fraction': changed / states,
            'opposite_length_fraction': switched / states, 'action_counts': counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    args = parser.parse_args()
    report = {'protocol': '30 matched synthetic states; only current-batter memory changes; not a performance evaluation',
              'results': {name: probe(name) for name in ('varying', 'fixed:3')}}
    for path in sorted(args.study.glob('*/ppo.zip')):
        report['results'][path.parent.name] = probe(load_model(path))
    (args.study / 'history-probes.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: value['history_changed_action_fraction'] for name, value in report['results'].items()}, indent=2))


if __name__ == '__main__':
    main()
