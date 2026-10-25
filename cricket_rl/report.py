"""Render measured multi-seed results as a standalone report and SVG chart."""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import numpy as np


def interval(values):
    """Student-t interval over independent training-seed means (df 1..30)."""
    n = len(values)
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    # Conservative df=10 critical for larger samples; avoid understated intervals.
    t = critical.get(n - 1, 2.228)
    return float(np.mean(values)), float(t * np.std(values, ddof=1) / np.sqrt(n))


def aggregate(study):
    rows = []
    for scenario, baselines in study['baselines'].items():
        for policy, result in baselines.items():
            s = result['summary']
            rows.append(dict(scenario=scenario, policy='constant' if policy.startswith('fixed:') else policy,
                             reward=s['reward']['mean'], ci=s['reward']['ci95_half_width'],
                             economy=s['runs_per_over'], wickets=s['wickets']['mean'],
                             dominant_action=s.get('dominant_action_fraction')))
        for mode in ('single', 'mixed'):
            results = [r[scenario]['summary'] for name, r in study['trained'].items()
                       if name.startswith(mode + '-')]
            if len(results) < 2:
                continue
            reward, ci = interval([r['reward']['mean'] for r in results])
            rows.append(dict(scenario=scenario, policy='PPO ' + mode, reward=reward, ci=ci,
                             economy=float(np.mean([r['runs_per_over'] for r in results])),
                             wickets=float(np.mean([r['wickets']['mean'] for r in results])),
                             dominant_action=float(np.mean([r.get('dominant_action_fraction', 0) for r in results]))))
    return rows


def learning_chart(directory, output):
    curves = {mode: [] for mode in ('single', 'mixed')}
    for mode in curves:
        for path in sorted(Path(directory).glob(f'{mode}-*/training.monitor.csv')):
            with path.open() as stream:
                rows = list(csv.DictReader(line for line in stream if not line.startswith('#')))
            if len(rows) < 20:
                continue
            steps = np.cumsum([int(r['l']) for r in rows])
            rewards = np.array([float(r['r']) for r in rows])
            window = min(50, len(rows))
            smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
            curves[mode].append((steps[window-1:], smooth))
    all_curves = [curve for group in curves.values() for curve in group]
    if not all_curves:
        return ''
    xmin = max(x[0] for x, _ in all_curves)
    xmax = min(x[-1] for x, _ in all_curves)
    if xmax <= xmin:
        return ''
    grid = np.linspace(xmin, xmax, 100)
    means = {mode: np.mean([np.interp(grid, x, y) for x, y in group], axis=0)
             for mode, group in curves.items() if group}
    low = min(y.min() for y in means.values()) - 2
    high = max(y.max() for y in means.values()) + 2
    items = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 300" role="img" aria-label="Training reward curves">']
    for index, (mode, values) in enumerate(means.items()):
        points = ' '.join(f'{70+(x-xmin)/(xmax-xmin)*870:.1f},{240-(y-low)/(high-low)*210:.1f}' for x,y in zip(grid,values))
        color = '#7bc2ff' if mode == 'single' else '#b6f786'
        items.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        items.append(f'<text x="{70+index*300}" y="20" fill="{color}" font-size="14">PPO {mode}</text>')
    items.append(f'<text x="70" y="275" fill="#dce9ee" font-size="12">{xmin:.0f} → {xmax:.0f} training steps · reward: {low:.1f} to {high:.1f} · 50-episode smoothing, mean across seeds</text></svg>')
    chart = ''.join(items)
    Path(output).write_text(chart)
    return chart


def generate(source, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    study = json.loads(Path(source).read_text())
    rows = aggregate(study)
    colors = {'random': '#b9c6cc', 'heuristic': '#ffd284',
              'varying': '#ff8d9a', 'constant': '#e39af0', 'PPO single': '#7bc2ff', 'PPO mixed': '#b6f786'}
    width, row_height = 1000, 32
    low = min(r['reward'] - r['ci'] for r in rows) - 2
    high = max(r['reward'] + r['ci'] for r in rows) + 2
    scale = lambda value: 280 + (value - low) / max(high - low, 1) * 650
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {len(rows)*row_height+70}" role="img" aria-label="Evaluation reward and uncertainty by policy">']
    for i, row in enumerate(rows):
        y = 40 + i * row_height
        x, left, right = [scale(v) for v in [row['reward'], row['reward']-row['ci'], row['reward']+row['ci']]]
        label = escape(row['scenario'] + ' / ' + row['policy'])
        svg.extend([f'<text x="5" y="{y+4}" fill="#dce9ee" font-size="12">{label}</text>',
                    f'<path d="M{left},{y} H{right}" stroke="{colors[row["policy"]]}" stroke-width="2"/>',
                    f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[row["policy"]]}"/>'])
    svg.append(f'<text x="280" y="{len(rows)*row_height+60}" fill="#dce9ee" font-size="12">Mean bowling reward (higher is better); horizontal intervals: approximate 95%</text></svg>')
    chart = ''.join(svg)
    table = ''.join(f'<tr><td>{escape(r["scenario"])}</td><td>{r["policy"]}</td><td>{r["reward"]:.2f} ± {r["ci"]:.2f}</td><td>{r["economy"]:.2f}</td><td>{r["wickets"]:.2f}</td><td>{r.get("dominant_action",0) or 0:.1%}</td></tr>' for r in rows)
    config = study['config']
    learning = learning_chart(Path(source).parent, Path(output).with_name('learning.svg'))
    text = f'''<!doctype html><meta charset="utf-8"><title>Cricket Strategy Lab — experiment report</title>
<style>body{{background:#0a141c;color:#e9f3f1;font:15px system-ui;max-width:1100px;margin:40px auto;padding:20px}}h1{{color:#b6f786}}svg{{width:100%}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #24414d}}p{{line-height:1.7;color:#b6cad2}}</style>
<h1>Cricket strategy generalization study</h1>
<p>Measured local experiment: {len(config['seeds'])} training seeds per condition, {config['steps']:,} requested PPO steps per seed, {config['episodes']} evaluation innings per scenario and policy. Single-style and mixed-style agents share all hyperparameters and train on the standard surface. Held-out pitch conditions test transfer without pitch training. The static-control scenario disables adaptation; it is an ablation, not an additional held-out opponent.</p>
{chart}<h2>Training reward</h2>{learning}<p>Training curves are descriptive smoothed means, not convergence evidence or confidence intervals.</p><table><thead><tr><th>Scenario</th><th>Policy</th><th>Reward ± interval</th><th>Runs / over ↓</th><th>Wickets / innings ↑</th><th>Dominant action</th></tr></thead><tbody>{table}</tbody></table>
<p>PPO intervals use Student-t uncertainty over training-seed mean rewards. Baseline intervals are normal approximations over evaluation episodes. These estimate different uncertainty sources and must not be treated as a significance test. A zero training-seed interval can arise when every trained agent repeats the same action; it does not mean evaluation uncertainty is zero. Fixed evaluation seeds are shared across policies; delivery noise diverges with action choices. The reward is 8 × wickets − runs. All results concern this uncalibrated simulator; no real-world or SOTA claim follows.</p>
<p>Inspect study.json for episode-level data, truncations, complete configuration, source hashes, and package versions. Each checkpoint has a training monitor CSV and metadata. The limited training budget is a pilot experiment, not a convergence claim.</p>'''
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    output.with_suffix('.svg').write_text(chart)
    output.with_suffix('.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/report.html'))
    args = parser.parse_args()
    generate(args.source, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
