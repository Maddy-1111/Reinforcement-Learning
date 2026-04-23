"""Plot LunarLander learning curves.

Supports:
    - continuous vanilla    (Q2.2.1/2)
    - continuous hover-box, pre/post reward swap overlay (Q2.2.3)
    - discrete SAC vs DQN   (Q2.2.4(c))

Run `python plot.py --help` for options.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load(path):
    steps, means, stds = [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row['step']))
            means.append(float(row['return_mean']))
            stds.append(float(row['return_std']))
    return np.asarray(steps), np.asarray(means), np.asarray(stds)


def agg(paths):
    curves = [load(p) for p in paths if p.exists()]
    if not curves:
        return None
    min_len = min(len(c[0]) for c in curves)
    steps = curves[0][0][:min_len]
    arr = np.stack([c[1][:min_len] for c in curves], axis=0)
    return steps, arr.mean(0), arr.std(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-dir', default='runs')
    p.add_argument('--seeds', nargs='+', type=int, default=[1])
    p.add_argument('--out', default='lander_curves.png')
    p.add_argument('--mode', choices=['continuous', 'hover', 'discrete-vs-dqn'],
                   default='continuous')
    p.add_argument('--swap-step', type=int, default=None,
                   help='Mark the hover-bonus swap step on the figure.')
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    fig, ax = plt.subplots(figsize=(9, 6))

    if args.mode == 'continuous':
        for mode in ('auto', 'manual'):
            paths = [runs_dir / f"sac_{mode}_seed{s}" / 'progress.csv'
                     for s in args.seeds]
            res = agg(paths)
            if res is None:
                continue
            s, m, sd = res
            ax.plot(s, m, label=f"SAC ({mode})")
            ax.fill_between(s, m - sd, m + sd, alpha=0.2)
        ax.set_title('LunarLander continuous — SAC')

    elif args.mode == 'hover':
        # Look for any run dir with 'hov' in it.
        for run_dir in sorted(runs_dir.glob('*hov*')):
            paths = []
            for s in args.seeds:
                p2 = run_dir.with_name(run_dir.name.replace(
                    f'_seed{args.seeds[0]}', f'_seed{s}')) / 'progress.csv'
                if p2.exists():
                    paths.append(p2)
            res = agg(paths or [run_dir / 'progress.csv'])
            if res is None:
                continue
            s, m, sd = res
            ax.plot(s, m, label=run_dir.name)
            ax.fill_between(s, m - sd, m + sd, alpha=0.2)
        if args.swap_step is not None:
            ax.axvline(args.swap_step, color='k', linestyle='--',
                       alpha=0.5, label='reward swap')
        ax.set_title('LunarLander hover-box — before/after reward swap')

    elif args.mode == 'discrete-vs-dqn':
        for label, prefix in [('discrete-SAC', 'sac_discrete'),
                               ('DQN', 'dqn')]:
            paths = [runs_dir / f"{prefix}_seed{s}" / 'progress.csv'
                     for s in args.seeds]
            res = agg(paths)
            if res is None:
                continue
            s, m, sd = res
            ax.plot(s, m, label=label)
            ax.fill_between(s, m - sd, m + sd, alpha=0.2)
        ax.set_title('LunarLander discrete — SAC vs DQN')

    ax.set_xlabel('env steps')
    ax.set_ylabel('mean return (20 eps)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
