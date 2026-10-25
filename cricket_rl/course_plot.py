"""Plot actual course-study results and per-seed learning curves (analysis extra)."""
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('study',type=Path)
    p.add_argument('--output',type=Path,default=Path('runs/course-plots'))
    args=p.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'cricket-mpl'))
    os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'cricket-plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((args.study/'summary.json').read_text())
    algorithms=['qlearning','ppo','dqn']
    colors={'qlearning':'#b27616','ppo':'#2678b2','dqn':'#398636'}
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,role in zip(axes,['batting','bowling']):
        for offset,scenario in [(-.12,'familiar'),(.12,'heldout')]:
            subset=[next(r for r in rows if r['role']==role and r['algorithm']==algo and r['scenario']==scenario) for algo in algorithms]
            ax.errorbar(np.arange(3)+offset,[r['reward'] for r in subset],yerr=[r['ci95'] for r in subset],fmt='o',capsize=5,label=scenario)
        ax.set(xticks=np.arange(3),xticklabels=algorithms,title=role.title(),ylabel='Episode reward (higher is better)')
        ax.grid(axis='y',alpha=.25)
        ax.legend()
    fig.suptitle('Equal transition budgets; 95% intervals over three training-seed means')
    for suffix in ['svg','png']:fig.savefig(args.output/f'comparison.{suffix}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,role in zip(axes,['batting','bowling']):
        seen=set()
        for folder in sorted(args.study.glob(f'{role}-*')):
            if not folder.is_dir():continue
            algo=folder.name.split('-')[1]
            if (folder/'learning.json').exists():points=json.loads((folder/'learning.json').read_text())
            else:
                with (folder/'training.monitor.csv').open() as f:
                    records=list(csv.DictReader(line for line in f if not line.startswith('#')))
                steps=np.cumsum([int(r['l']) for r in records])
                points=[{'step':int(s),'reward':float(r['r'])} for s,r in zip(steps,records)]
            if not points:continue
            window=min(25,len(points))
            reward=np.convolve([r['reward'] for r in points],np.ones(window)/window,mode='valid')
            steps=[r['step'] for r in points][window-1:]
            ax.plot(steps,reward,color=colors[algo],alpha=.6,linewidth=1,label=algo if algo not in seen else None)
            seen.add(algo)
        ax.set(title=role.title(),xlabel='Environment transitions',ylabel='25-episode mean reward')
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Every training seed shown; smoothing does not demonstrate convergence')
    for suffix in ['svg','png']:fig.savefig(args.output/f'learning.{suffix}',dpi=160)
    plt.close(fig)
    print(args.output)


if __name__=='__main__':main()
