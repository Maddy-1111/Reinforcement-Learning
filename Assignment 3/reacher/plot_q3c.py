"""Q2.3.3(c): 3 separate figures, one per training reward, each showing the
trained policy evaluated against R_a, R_b, R_c (all 3 lines per figure).

Mean ± 95% CI across seeds.

Usage:
    python plot_q3c.py --runs-dir logs --seeds 1 2 3 --smooth 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_q2 import collect, smooth  # reuse helpers


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-dir', default='logs')
    p.add_argument('--seeds', nargs='+', type=int, default=[1])
    p.add_argument('--smooth', type=int, default=1)
    p.add_argument('--out', default='q3c_perrow.png')
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    eval_variants = ['a', 'b', 'c']
    colors = {'a': 'C0', 'b': 'C1', 'c': 'C2'}

    for ax, train_r in zip(axes, ['a', 'b', 'c']):
        any_data = False
        for ev in eval_variants:
            steps, arr = collect(runs_dir, train_r, args.seeds, f'R_{ev}_mean')
            if steps is None:
                continue
            any_data = True
            mean = arr.mean(axis=0)
            if arr.shape[0] > 1:
                sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                ci = 1.96 * sem
            else:
                ci = np.zeros_like(mean)
            if args.smooth > 1:
                mean_s = smooth(mean, args.smooth)
                ci_s = smooth(ci, args.smooth)
                x = steps[len(steps) - len(mean_s):]
            else:
                mean_s, ci_s, x = mean, ci, steps
            ax.plot(x, mean_s, label=f'eval R{ev}', color=colors[ev])
            ax.fill_between(x, mean_s - ci_s, mean_s + ci_s,
                            alpha=0.20, color=colors[ev])
        if not any_data:
            ax.set_title(f"SAC-R{train_r} (no runs)")
            continue
        ax.set_title(f"Trained on R{train_r}")
        ax.set_xlabel('env steps')
        ax.set_ylabel('mean return')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    fig.suptitle(f"Q2.3.3(c) — cross-reward evaluation (seeds={args.seeds})",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
