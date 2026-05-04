"""Plot learning curves for Pendulum SAC runs (auto vs manual alpha).

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
    p.add_argument('--runs-dir', default='runs_cosine')
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
    p.add_argument('--alpha-sweep', action='store_true',
                   help='Sweep --alphas at single --theta (manual mode).')
    p.add_argument('--alphas', nargs='+', type=float,
                   default=[0.005, 0.01, 0.015, 0.02, 0.03, 0.07])
    p.add_argument('--rs-sweep', action='store_true',
                   help='Sweep --scales at single --theta in --alpha-mode.')
    p.add_argument('--auto-dir', default=None,
                   help='Override --runs-dir for auto-mode runs (compare-modes).')
    p.add_argument('--manual-dir', default=None,
                   help='Override --runs-dir for manual-mode runs (compare-modes).')
    p.add_argument('--compare-modes-multi', action='store_true',
                   help='auto vs manual for multiple thetas on one plot. '
                        'Pair --thetas with --manual-alphas (same length).')
    p.add_argument('--manual-alphas', nargs='+', type=float, default=None,
                   help='Per-theta manual alpha (aligned with --thetas).')
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    auto_dir = Path(args.auto_dir) if args.auto_dir else runs_dir
    manual_dir = Path(args.manual_dir) if args.manual_dir else runs_dir

    def run_paths(theta, mode, alpha=None, rs=1.0, seeds=None, base=None):
        seeds = seeds or args.seeds
        base = base if base is not None else runs_dir
        tag = f"theta{int(theta)}_{mode}"
        if mode == 'manual':
            tag += f"_a{alpha:g}"
        if rs != 1.0:
            tag += f"_rs{rs:g}"
        return [base / f"sac_{tag}_seed{s}" / 'progress.csv' for s in seeds]

    fig, ax = plt.subplots(figsize=(9, 6))

    if args.compare_modes_multi:
        if not args.manual_alphas or len(args.manual_alphas) != len(args.thetas):
            raise SystemExit('--compare-modes-multi requires --manual-alphas '
                             'matching --thetas length.')
        cmap = plt.get_cmap('tab10')
        for i, (th, a) in enumerate(zip(args.thetas, args.manual_alphas)):
            color = cmap(i % 10)
            res_a = agg(run_paths(th, 'auto', base=auto_dir))
            if res_a is not None:
                s, m, sd = res_a
                ax.plot(s, m, color=color, linestyle='-',
                        label=f"theta={int(th)} auto")
                ax.fill_between(s, m - sd, m + sd, color=color, alpha=0.15)
            res_m = agg(run_paths(th, 'manual', alpha=a, base=manual_dir))
            if res_m is not None:
                s, m, sd = res_m
                ax.plot(s, m, color=color, linestyle='--',
                        label=f"theta={int(th)} manual a={a:g}")
                ax.fill_between(s, m - sd, m + sd, color=color, alpha=0.15)
        ax.set_title("Pendulum: auto vs manual alpha across thetas")
    elif args.compare_modes:
        for mode, alpha, base in [
            ('auto', None, auto_dir),
            ('manual', args.manual_alpha, manual_dir),
        ]:
            res = agg(run_paths(args.theta, mode, alpha=alpha, rs=args.rs,
                                base=base))
            if res is None:
                continue
            s, m, sd = res
            lbl = f"{mode}" + (f" (a={alpha:g})" if alpha is not None else "")
            ax.plot(s, m, label=lbl)
            ax.fill_between(s, m - sd, m + sd, alpha=0.2)
        ax.set_title(f"Pendulum theta={args.theta}: auto vs manual alpha")
    elif args.alpha_sweep:
        for a in args.alphas:
            res = agg(run_paths(args.theta, 'manual', alpha=a))
            if res is None:
                continue
            s, m, sd = res
            ax.plot(s, m, label=f"alpha={a:g}")
            ax.fill_between(s, m - sd, m + sd, alpha=0.15)
        ax.set_title(f"Pendulum theta={args.theta}: manual alpha sweep")
    elif args.rs_sweep:
        mode = args.alpha_mode
        for rs in args.scales:
            res = agg(run_paths(args.theta, mode,
                                alpha=args.manual_alpha if mode == 'manual' else None,
                                rs=rs))
            if res is None:
                continue
            s, m, sd = res
            ax.plot(s, m, label=f"rs={rs:g}")
            ax.fill_between(s, m - sd, m + sd, alpha=0.15)
        ax.set_title(f"Pendulum theta={args.theta} ({mode}): reward-scale sweep")
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

    ax.set_ylim(-2500, 1000)                      # <-- IMPORTANT
    ax.set_yticks(np.arange(-2500, 1001, 200))    # tick spacing

    ax.set_xlim(0, 50_000)
    ax.set_xticks(np.arange(0, 50_001, 10_000))

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
