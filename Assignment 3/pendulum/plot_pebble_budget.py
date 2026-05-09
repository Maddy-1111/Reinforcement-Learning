"""Plot PEBBLE feedback-budget sweep for one theta (mean ± seed-std)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load(path: Path):
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
    means = np.stack([c[1][:min_len] for c in curves], axis=0)
    return steps, means.mean(0), means.std(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--theta', type=float, required=True)
    p.add_argument('--pebble-dir', default='pebble')
    p.add_argument('--budgets', nargs='+', type=int, required=True)
    p.add_argument('--seeds', nargs='+', type=int, required=True,
                   help='Seed pool. Missing run dirs are silently skipped.')
    p.add_argument('--selection', default='disagreement')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap('tab10')

    for i, fb in enumerate(args.budgets):
        paths = [
            Path(args.pebble_dir) /
            f"pebble_theta{int(args.theta)}_fb{fb}_{args.selection}_seed{s}" /
            'progress.csv'
            for s in args.seeds
        ]
        res = agg(paths)
        if res is None:
            continue
        s, m, sd = res
        color = cmap(i % 10)
        ax.plot(s, m, color=color, label=f"PEBBLE fb={fb}")
        ax.fill_between(s, m - sd, m + sd, color=color, alpha=0.2)

    ax.set_xlabel('env steps')
    ax.set_ylabel('mean return (20 eps, ground-truth reward)')
    ax.set_title(f"Pendulum theta={int(args.theta)}: PEBBLE feedback-budget sweep")
    ax.set_ylim(-2000, 1000)
    ax.set_yticks(np.arange(-2000, 1001, 200))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
