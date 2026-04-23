"""Plot cross-reward evaluation curves for the three SAC-R{a,b,c} runs.

Produces a 3x3 grid: rows = training reward, cols = eval reward. Matches the
figure format expected by the assignment.

Usage:
    python plot_results.py --runs-dir runs
    python plot_results.py --runs-dir runs --seeds 1 2 3 --smooth 5
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_progress(csv_path):
    steps, rets = [], {'a': [], 'b': [], 'c': []}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            rets['a'].append(float(row['R_a_mean']))
            rets['b'].append(float(row['R_b_mean']))
            rets['c'].append(float(row['R_c_mean']))
    return np.array(steps), {k: np.array(v) for k, v in rets.items()}


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode='valid')


def find_runs(runs_dir, train_r, seeds):
    paths = []
    for seed in seeds:
        p = runs_dir / f"sac_R{train_r}_seed{seed}" / "progress.csv"
        if p.exists():
            paths.append((seed, p))
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-dir', default='runs')
    p.add_argument('--seeds', nargs='+', type=int, default=[1])
    p.add_argument('--smooth', type=int, default=1)
    p.add_argument('--out', default='cross_reward_eval.png')
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    train_variants = ['a', 'b', 'c']
    eval_variants = ['a', 'b', 'c']

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)

    for i, trained in enumerate(train_variants):
        run_paths = find_runs(runs_dir, trained, args.seeds)
        if not run_paths:
            for j in range(3):
                axes[i, j].set_title(f"(no runs for SAC-R{trained})")
            continue
        # collect per-seed curves aligned on shortest length
        seed_curves = {v: [] for v in eval_variants}
        step_ref = None
        for seed, path in run_paths:
            steps, rets = load_progress(path)
            if step_ref is None or len(steps) < len(step_ref):
                step_ref = steps
            for v in eval_variants:
                seed_curves[v].append(rets[v])
        # trim to min length
        min_len = min(len(x) for curves in seed_curves.values() for x in curves)
        step_ref = step_ref[:min_len]
        for j, ev in enumerate(eval_variants):
            arr = np.stack([c[:min_len] for c in seed_curves[ev]], axis=0)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            if args.smooth > 1:
                mean_s = smooth(mean, args.smooth)
                std_s = smooth(std, args.smooth)
                x = step_ref[len(step_ref) - len(mean_s):]
            else:
                mean_s, std_s, x = mean, std, step_ref
            ax = axes[i, j]
            ax.plot(x, mean_s, label=f"SAC-R{trained}")
            ax.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.25)
            ax.set_title(f"Trained R{trained} | Eval R{ev}")
            ax.grid(True, alpha=0.3)
            if i == 2:
                ax.set_xlabel('env steps')
            if j == 0:
                ax.set_ylabel('mean return (20 eps)')

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
