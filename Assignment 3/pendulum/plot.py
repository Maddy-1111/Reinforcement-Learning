"""Plot learning curves for Pendulum SAC runs (auto vs manual alpha).

Expects runs under --runs-dir with names like
    sac_theta{deg}_auto_seed{N}
    sac_theta{deg}_manual_a{alpha}_seed{N}
    sac_theta{deg}_manual_a{alpha}_rs{scale}_seed{N}

Usage:
    # Q2.1.2: all theta_target curves with automated alpha
    python plot.py --runs-dir runs --alpha-mode auto --seeds 1 2 3 \\
        --thetas 0 -10 30 -60 90 -90 120 -150 --out auto_sweep.png

    # Q2.1.5a: auto vs manual at a given theta
    python plot.py --runs-dir runs --compare-modes --theta 90 \\
        --manual-alpha 0.2 --seeds 1 2 3 --out auto_vs_manual_90.png

    # Q2.1.5b: reward scale
    python plot.py --runs-dir runs --reward-scale-sweep --theta 90 \\
        --manual-alpha 0.2 --seeds 1 2 3 --out reward_scale_90.png
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
    means = np.stack([c[1][:min_len] for c in curves], axis=0)
    return steps, means.mean(0), means.std(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-dir', default='runs')
    p.add_argument('--seeds', nargs='+', type=int, default=[1])
    p.add_argument('--out', default='pendulum_curves.png')

    p.add_argument('--alpha-mode', choices=['auto', 'manual'], default='auto')
    p.add_argument('--thetas', nargs='+', type=float,
                   default=[0, -10, 30, -60, 90, -90, 120, -150])
    p.add_argument('--manual-alpha', type=float, default=None)

    p.add_argument('--compare-modes', action='store_true',
                   help='auto vs manual at single theta')
    p.add_argument('--theta', type=float, default=90.0)
    p.add_argument('--reward-scale-sweep', action='store_true')
    p.add_argument('--scales', nargs='+', type=float, default=[0.1, 1.0, 10.0])
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)

    def run_paths(theta, mode, alpha=None, rs=1.0, seeds=None):
        seeds = seeds or args.seeds
        tag = f"theta{int(theta)}_{mode}"
        if mode == 'manual':
            tag += f"_a{alpha:g}"
        if rs != 1.0:
            tag += f"_rs{rs:g}"
        return [runs_dir / f"sac_{tag}_seed{s}" / 'progress.csv' for s in seeds]

    fig, ax = plt.subplots(figsize=(9, 6))

    if args.compare_modes:
        for mode, alpha in [('auto', None), ('manual', args.manual_alpha)]:
            res = agg(run_paths(args.theta, mode, alpha=alpha))
            if res is None:
                continue
            s, m, sd = res
            lbl = f"{mode}" + (f" (a={alpha:g})" if alpha is not None else "")
            ax.plot(s, m, label=lbl)
            ax.fill_between(s, m - sd, m + sd, alpha=0.2)
        ax.set_title(f"Pendulum theta={args.theta}: auto vs manual alpha")
    elif args.reward_scale_sweep:
        for rs in args.scales:
            for mode, alpha in [('auto', None), ('manual', args.manual_alpha)]:
                if mode == 'manual' and alpha is None:
                    continue
                res = agg(run_paths(args.theta, mode, alpha=alpha, rs=rs))
                if res is None:
                    continue
                s, m, sd = res
                lbl = f"rs={rs:g} {mode}" + (
                    f" (a={alpha:g})" if alpha is not None else "")
                ax.plot(s, m, label=lbl)
                ax.fill_between(s, m - sd, m + sd, alpha=0.15)
        ax.set_title(f"Pendulum theta={args.theta}: reward scaling")
    else:
        for th in args.thetas:
            res = agg(run_paths(th, args.alpha_mode,
                                alpha=args.manual_alpha if args.alpha_mode == 'manual' else None))
            if res is None:
                continue
            s, m, sd = res
            ax.plot(s, m, label=f"theta={int(th)}")
            ax.fill_between(s, m - sd, m + sd, alpha=0.15)
        ax.set_title(f"Pendulum ({args.alpha_mode} alpha): theta sweep")

    ax.set_xlabel('env steps')
    ax.set_ylabel('mean return (20 eps)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
