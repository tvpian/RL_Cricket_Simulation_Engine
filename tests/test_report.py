import json
import numpy as np
import pytest
from cricket_rl.experiment import summarize, evaluate
from cricket_rl.report import interval, generate


def test_uncertainty_uses_training_seed_variance():
    mean, half = interval([1, 2, 3, 4, 5])
    assert mean == 3
    assert half == pytest.approx(2.776 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5))


def test_report_contains_actual_computed_values(tmp_path):
    rows = evaluate('random', episodes=3)
    s = {'summary': summarize(rows), 'episodes': rows}
    study = {'config': {'steps': 256, 'seeds': [0, 1], 'episodes': 3},
             'baselines': {'heldout_batter': {'random': s, 'heuristic': s}},
             'trained': {f'{mode}-{seed}': {'heldout_batter': s}
                         for mode in ('single', 'mixed') for seed in (0, 1)}}
    source = tmp_path / 'study.json'
    source.write_text(json.dumps(study))
    output = tmp_path / 'report.html'
    results = generate(source, output)
    assert len(results) == 4
    assert results[2]['ci'] == 0  # identical training-seed rewards
    assert 'PPO mixed' in output.read_text()
    assert output.with_suffix('.svg').exists()
    assert json.loads(output.with_suffix('.json').read_text()) == results
