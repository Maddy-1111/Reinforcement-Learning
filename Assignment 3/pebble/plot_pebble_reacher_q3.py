"""Bonus §3 Q3: PEBBLE-on-Reacher learning curves for the three teacher types.

One figure with three lines (teacher = R_a, R_b, R_c). Each line is
mean ± 95% CI across seeds. y-axis is the ground-truth (teacher) return
logged in PEBBLE's progress.csv (`return_mean` column).

Optionally overlays the matching SAC-R_i curves (trained with the true reward,
not learned) as dashed lines, for the "PEBBLE vs SAC ground-truth" comparison
the question asks for.

Usage:
    # PEBBLE only
    python plot_pebble_reacher_q3.py --runs-dir runs_pebble --seeds 1 2 3 --smooth 3

    # With SAC ground-truth overlay
    python plot_pebble_reacher_q3.py --runs-dir runs_pebble --seeds 1 2 3 \
        --sac-runs-dir ../reacher/logs --sac-seeds 1 2 3 4 5 --smooth 3
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path: Path):
    """Return dict of column -> np.array of floats; 'step' as int array."""
    out = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                if v == '':
                    continue
                try:
                    val = float(v)
                except ValueError:
                    continue
                out.setdefault(k, []).append(val)
    return {k: np.asarray(v) for k, v in out.items()}


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode='valid')


def find_pebble_run(runs_dir: Path, teacher: str, seed: int):
    """PEBBLE log dir: pebble_reacher_R{teacher}_fb*_*_seed{seed}.
    Pick the first match (typical case: one config per seed)."""
    matches = sorted(runs_dir.glob(
        f"pebble_reacher_R{teacher}_fb*_*_seed{seed}"))
    return matches[0] / "progress.csv" if matches else None


def collect_pebble(runs_dir: Path, teacher: str, seeds):
    """Return (steps, [n_seeds, n_steps] array of return_mean), trimmed."""
    curves, step_ref = [], None
    for s in seeds:
        p = find_pebble_run(runs_dir, teacher, s)
        if p is None or not p.exists():
            continue
        d = load_csv(p)
        if 'return_mean' not in d or 'step' not in d:
            continue
        steps = d['step'].astype(int)
        if step_ref is None or len(steps) < len(step_ref):
            step_ref = steps
        curves.append(d['return_mean'])
    if not curves:
        return None, None
    n = min(len(c) for c in curves)
    return step_ref[:n], np.stack([c[:n] for c in curves], axis=0)


def collect_sac(runs_dir: Path, train_r: str, seeds):
    """SAC-R{i} curve evaluated under R{i} (the diagonal)."""
    col = f'R_{train_r}_mean'
    curves, step_ref = [], None
    for s in seeds:
        p = runs_dir / f"sac_R{train_r}_seed{s}" / "progress.csv"
        if not p.exists():
            continue
        d = load_csv(p)
        if col not in d or 'step' not in d:
            continue
        steps = d['step'].astype(int)
        if step_ref is None or len(steps) < len(step_ref):
            step_ref = steps
        curves.append(d[col])
    if not curves:
        return None, None
    n = min(len(c) for c in curves)
    return step_ref[:n], np.stack([c[:n] for c in curves], axis=0)


def mean_ci95(arr):
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        return mean, 1.96 * sem
    return mean, np.zeros_like(mean)


def plot_with_band(ax, x, mean, ci, label, color, smooth_w, linestyle='-'):
    if smooth_w > 1:
        mean = smooth(mean, smooth_w)
        ci = smooth(ci, smooth_w)
        x = x[len(x) - len(mean):]
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle)
    ax.fill_between(x, mean - ci, mean + ci, alpha=0.20, color=color)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-dir', default='runs_pebble',
                   help='Dir containing pebble_reacher_R*_seed* subdirs')
    p.add_argument('--seeds', nargs='+', type=int, default=[1])
    p.add_argument('--sac-runs-dir', default=None,
                   help='Optional: overlay SAC-R_i diagonal curves '
                        'from this dir (containing sac_R*_seed* subdirs)')
    p.add_argument('--sac-seeds', nargs='+', type=int, default=None)
    p.add_argument('--smooth', type=int, default=1)
    p.add_argument('--out', default='bonus_q3_pebble.png')
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    teacher_color = {'a': 'C0', 'b': 'C1', 'c': 'C2'}

    for ax, teacher in zip(axes, ['a', 'b', 'c']):
        # PEBBLE curve
        steps, arr = collect_pebble(runs_dir, teacher, args.seeds)
        if steps is None:
            ax.set_title(f"PEBBLE teacher=R{teacher} (no runs)")
            continue
        mean, ci = mean_ci95(arr)
        plot_with_band(ax, steps, mean, ci,
                       f'PEBBLE-R{teacher}', teacher_color[teacher],
                       args.smooth)

        # Optional SAC ground-truth overlay
        if args.sac_runs_dir is not None:
            sac_seeds = args.sac_seeds or args.seeds
            sac_steps, sac_arr = collect_sac(
                Path(args.sac_runs_dir), teacher, sac_seeds)
            if sac_steps is not None:
                sac_mean, sac_ci = mean_ci95(sac_arr)
                plot_with_band(ax, sac_steps, sac_mean, sac_ci,
                               f'SAC-R{teacher} (ground truth)',
                               teacher_color[teacher],
                               args.smooth, linestyle='--')

        ax.set_title(f"Teacher = R{teacher}")
        ax.set_xlabel('env steps')
        ax.set_ylabel(f'return under R{teacher}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    fig.suptitle(
        f"Bonus §3 Q3 — PEBBLE on Reacher (seeds={args.seeds})",
        y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
