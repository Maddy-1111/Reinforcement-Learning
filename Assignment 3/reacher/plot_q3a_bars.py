"""Q2.3.3(a): Bar chart of `steps_to_goal` and `steps_in_target` for the three
final policies (SAC-R{a,b,c}). CIs over the 500 evaluation episodes (and across
seeds if multiple).

Reads `final_eval.csv` from each run dir (produced by `eval_final_policy.py`).

Usage:
    python plot_q3a_bars.py --runs-dir logs --seeds 1 2 3 \
        --max-steps 5000 --out q3a_bars.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_final_eval(csv_path: Path):
    stg, sit = [], []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            s = row['steps_to_goal']
            stg.append(float('inf') if s in ('inf', 'Infinity') else float(s))
            sit.append(int(float(row['steps_in_target'])))
    return np.array(stg), np.array(sit)


def collect_episodes(runs_dir: Path, train_r: str, seeds, max_steps: int):
    """Pool steps_to_goal and steps_in_target across all episodes of all seeds.
    `steps_to_goal` is clipped at `max_steps` (treats never-reached as worst-case).
    Returns (stg_clipped, sit, n_total, n_reached)."""
    stg_all, sit_all = [], []
    for s in seeds:
        p = runs_dir / f"sac_R{train_r}_seed{s}" / "final_eval.csv"
        if not p.exists():
            continue
        stg, sit = load_final_eval(p)
        stg_all.append(stg)
        sit_all.append(sit)
    if not stg_all:
        return None, None, 0, 0
    stg = np.concatenate(stg_all)
    sit = np.concatenate(sit_all)
    n_total = len(stg)
    n_reached = int(np.isfinite(stg).sum())
    stg_clipped = np.where(np.isfinite(stg), stg, float(max_steps))
    return stg_clipped, sit, n_total, n_reached


def ci95(x):
    if len(x) < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-dir', default='logs')
    p.add_argument('--seeds', nargs='+', type=int, default=[1])
    p.add_argument('--max-steps', type=int, default=5000,
                   help='Episode length used by eval_final_policy.py; also '
                        'used to clip never-reached steps_to_goal.')
    p.add_argument('--out', default='q3a_bars.png')
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    rewards = ['a', 'b', 'c']
    stg_means, stg_cis = [], []
    sit_means, sit_cis = [], []
    reach_rates = []

    for r in rewards:
        stg, sit, n_total, n_reached = collect_episodes(
            runs_dir, r, args.seeds, args.max_steps)
        if stg is None:
            stg_means.append(np.nan); stg_cis.append(0.0)
            sit_means.append(np.nan); sit_cis.append(0.0)
            reach_rates.append(np.nan)
            print(f"SAC-R{r}: NO DATA")
            continue
        stg_means.append(float(stg.mean()))
        stg_cis.append(float(ci95(stg)))
        sit_means.append(float(sit.mean()))
        sit_cis.append(float(ci95(sit)))
        rr = n_reached / n_total
        reach_rates.append(rr)
        print(f"SAC-R{r}: n={n_total}  reached={n_reached} ({rr:.1%})  "
              f"steps_to_goal={stg.mean():.0f}±{ci95(stg):.0f}  "
              f"steps_in_target={sit.mean():.0f}±{ci95(sit):.0f}")

    x = np.arange(len(rewards))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(x, stg_means, yerr=stg_cis, capsize=6,
                color=['C0', 'C1', 'C2'])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"SAC-R{r}" for r in rewards])
    axes[0].set_ylabel('steps to goal (lower = better)')
    axes[0].set_title(f'Steps to reach target (cap={args.max_steps})')
    axes[0].grid(True, alpha=0.3, axis='y')
    # annotate reach-rate
    for xi, rr in zip(x, reach_rates):
        if not np.isnan(rr):
            axes[0].text(xi, axes[0].get_ylim()[1] * 0.92,
                         f"reached {rr:.0%}",
                         ha='center', fontsize=8, color='dimgray')

    axes[1].bar(x, sit_means, yerr=sit_cis, capsize=6,
                color=['C0', 'C1', 'C2'])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"SAC-R{r}" for r in rewards])
    axes[1].set_ylabel('steps in target (higher = better)')
    axes[1].set_title('Steps spent in target region')
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"Q2.3.3(a) — final-policy eval (500 eps, seeds={args.seeds})",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
