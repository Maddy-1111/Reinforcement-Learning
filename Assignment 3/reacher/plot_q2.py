"""Q2.3.2: 3 separate figures, one per training reward, showing SAC-R_i evaluated
against R_i (the diagonal of the cross-reward grid).

Each figure: x = env steps, y = mean undiscounted return over 20 eval eps,
mean ± 95% CI across seeds.

Usage:
    python plot_q2.py --runs-dir logs --seeds 1 2 3 --smooth 3
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_progress(csv_path):
    steps, cols = [], {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            for k, v in row.items():
                if k == 'step' or v == '':
                    continue
                cols.setdefault(k, []).append(float(v))
    return np.array(steps), {k: np.array(v) for k, v in cols.items()}


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode='valid')


def collect(runs_dir: Path, train_r: str, seeds, eval_key: str):
    """Return (steps, [n_seeds, n_steps] array) for a given training reward
    and eval column."""
    curves, step_ref = [], None
    for s in seeds:
        p = runs_dir / f"sac_R{train_r}_seed{s}" / "progress.csv"
        if not p.exists():
            continue
        steps, cols = load_progress(p)
        if eval_key not in cols:
            continue
        if step_ref is None or len(steps) < len(step_ref):
            step_ref = steps
        curves.append(cols[eval_key])
    if not curves:
        return None, None
    n = min(len(c) for c in curves)
    arr = np.stack([c[:n] for c in curves], axis=0)
    return step_ref[:n], arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-dir', default='logs')
    p.add_argument('--seeds', nargs='+', type=int, default=[1])
    p.add_argument('--smooth', type=int, default=1)
    p.add_argument('--out', default='q2_diagonal.png')
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, r in zip(axes, ['a', 'b', 'c']):
        steps, arr = collect(runs_dir, r, args.seeds, f'R_{r}_mean')
        if steps is None:
            ax.set_title(f"SAC-R{r} | R{r}  (no runs)")
            continue
        mean = arr.mean(axis=0)
        # 95% CI across seeds
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

        ax.plot(x, mean_s, label=f'SAC-R{r}', color='C0')
        ax.fill_between(x, mean_s - ci_s, mean_s + ci_s, alpha=0.25, color='C0')
        ax.set_title(f"SAC-R{r} evaluated under R{r}")
        ax.set_xlabel('env steps')
        ax.set_ylabel(f'mean return (R{r})')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Q2.3.2 — diagonal eval (seeds={args.seeds})", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
