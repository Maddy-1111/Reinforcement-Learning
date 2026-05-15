"""Plot LunarLander learning curves.

Modes:
    continuous      — Q2.2.1/2  (auto-α SAC, 15 seeds, 95% CI + seed traces)
    hover           — Q2.2.3    (manual vs auto α, reward-swap line)
    discrete-vs-dqn — Q2.2.4(c) (discrete-SAC vs DQN)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLE = {
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 11,
}
plt.rcParams.update(STYLE)

SEED_TRACE_KW  = dict(color='#aaaaaa', linewidth=0.6, alpha=0.45, zorder=1)
MEAN_LINEWIDTH = 2.2
CI_ALPHA       = 0.25


# ── data helpers ──────────────────────────────────────────────────────────────

def load(path: Path):
    """Return (steps, return_means, return_stds, alphas) arrays."""
    steps, means, stds, alphas = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row['step']))
            means.append(float(row['return_mean']))
            stds.append(float(row['return_std']))
            a = row.get('alpha', '')
            alphas.append(float(a) if a.strip() else float('nan'))
    return (np.asarray(steps), np.asarray(means),
            np.asarray(stds), np.asarray(alphas))


def agg(paths: list[Path]):
    """Aggregate seed curves → (steps, mean, 95%-CI-half, per-seed-means)."""
    curves = [(s, m, sd, a) for p in paths
              if p.exists() for s, m, sd, a in [load(p)]]
    if not curves:
        return None
    n = len(curves)
    L = min(len(c[0]) for c in curves)
    steps = curves[0][0][:L]
    arr   = np.stack([c[1][:L] for c in curves], axis=0)   # (n, T)
    alpha_arr = np.stack([c[3][:L] for c in curves], axis=0)
    mean  = arr.mean(0)
    ci    = 1.96 * arr.std(0) / np.sqrt(n)                 # 95 % CI
    alpha_mean = np.nanmean(alpha_arr, axis=0)
    alpha_ci   = 1.96 * np.nanstd(alpha_arr, axis=0) / np.sqrt(n)
    return steps, mean, ci, arr, alpha_mean, alpha_ci


def _add_seed_traces(ax, steps, per_seed_arr):
    for row in per_seed_arr:
        ax.plot(steps, row, **SEED_TRACE_KW)


def _add_curve(ax, steps, mean, ci, color, label, seed_arr=None):
    if seed_arr is not None:
        _add_seed_traces(ax, steps, seed_arr)
    ax.plot(steps, mean, color=color, linewidth=MEAN_LINEWIDTH,
            label=label, zorder=3)
    ax.fill_between(steps, mean - ci, mean + ci,
                    color=color, alpha=CI_ALPHA, zorder=2)


def _finalise(ax, title, out):
    ax.set_xlabel('Environment Timesteps', fontsize=12)
    ax.set_ylabel('Average Undiscounted Return', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')


# ── plot modes ────────────────────────────────────────────────────────────────

PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def plot_continuous(runs_dir, seeds, out):
    fig, ax = plt.subplots()
    paths = [runs_dir / f'sac_auto_seed{s}' / 'progress.csv' for s in seeds]
    res = agg(paths)
    if res:
        s, m, ci, arr, _, _ = res
        _add_curve(ax, s, m, ci, PALETTE[0], 'SAC (auto α)', seed_arr=arr)
    _finalise(ax, 'LunarLander Continuous — SAC (auto α, 95% CI)', out)


def plot_hover(runs_dir, seeds, swap_step, out):
    fig, ax = plt.subplots()
    variants = [
        ('sac_manual_a0.01_hov200_swap250000to-100', 'Fixed α = 0.01', PALETTE[0]),
        ('sac_auto_hov200_swap250000to-100',          'Auto α',         PALETTE[1]),
    ]
    for prefix, label, col in variants:
        paths = [runs_dir / f'{prefix}_seed{s}' / 'progress.csv' for s in seeds]
        res = agg(paths)
        if res:
            s, m, ci, arr, _, _ = res
            _add_curve(ax, s, m, ci, col, label, seed_arr=arr)
    if swap_step is not None:
        ax.axvline(swap_step, color='black', linestyle='--', linewidth=1.5,
                   alpha=0.7, label='Reward swap (+200 → −100)', zorder=4)
    _finalise(ax, 'LunarLander Hover-box — Fixed vs Auto α, Reward Swap (95% CI)', out)


def plot_discrete_vs_dqn(runs_dir, seeds, out):
    fig, ax = plt.subplots()
    variants = [
        ('sac_discrete', 'Discrete-SAC', PALETTE[0]),
        ('dqn',          'DQN',          PALETTE[1]),
    ]
    for prefix, label, col in variants:
        paths = [runs_dir / f'{prefix}_seed{s}' / 'progress.csv' for s in seeds]
        res = agg(paths)
        if res:
            s, m, ci, arr, _, _ = res
            _add_curve(ax, s, m, ci, col, label, seed_arr=arr)
    _finalise(ax, 'LunarLander Discrete — Discrete-SAC vs DQN (95% CI)', out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-dir', default='runs')
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 16)))
    p.add_argument('--out', default='lander_curves.png')
    p.add_argument('--mode',
                   choices=['continuous', 'hover', 'discrete-vs-dqn'],
                   default='continuous')
    p.add_argument('--swap-step', type=int, default=None)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    if args.mode == 'continuous':
        plot_continuous(runs_dir, args.seeds, args.out)
    elif args.mode == 'hover':
        plot_hover(runs_dir, args.seeds, args.swap_step, args.out)
    elif args.mode == 'discrete-vs-dqn':
        plot_discrete_vs_dqn(runs_dir, args.seeds, args.out)


if __name__ == '__main__':
    main()
