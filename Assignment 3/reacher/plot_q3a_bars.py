"""Q2.3.3(a): Bar chart of `steps_to_goal` and `steps_in_target` for the three
final policies (SAC-R{a,b,c}). CIs over the pooled evaluation episodes
(across seeds).

Reads a single combined CSV with columns: reward, seed, episode,
steps_to_goal, steps_in_target.

Usage:
    python plot_q3a_bars.py --csv eval_final.csv --max-steps 5000 --out q3a_bars.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_combined(csv_path: Path):
    """Return dict: reward -> {'stg': np.array, 'sit': np.array, 'seeds': set}."""
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            r = row['reward']
            d = out.setdefault(r, {'stg': [], 'sit': [], 'seeds': set()})
            s = row['steps_to_goal']
            stg = float('inf') if s in ('inf', 'Infinity') else float(s)
            d['stg'].append(stg)
            d['sit'].append(int(float(row['steps_in_target'])))
            d['seeds'].add(int(row['seed']))
    for r in out:
        out[r]['stg'] = np.array(out[r]['stg'])
        out[r]['sit'] = np.array(out[r]['sit'])
    return out


def ci95(x):
    if len(x) < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='eval_final.csv')
    p.add_argument('--max-steps', type=int, default=5000,
                   help='Episode length used by eval_final_policy.py; also '
                        'used to clip never-reached steps_to_goal.')
    p.add_argument('--out', default='q3a_bars.png')
    args = p.parse_args()

    data = load_combined(Path(args.csv))
    rewards = ['a', 'b', 'c']
    stg_means, stg_cis = [], []
    sit_means, sit_cis = [], []
    reach_rates = []

    for r in rewards:
        if r not in data:
            stg_means.append(np.nan); stg_cis.append(0.0)
            sit_means.append(np.nan); sit_cis.append(0.0)
            reach_rates.append(np.nan)
            print(f"SAC-R{r}: NO DATA")
            continue
        stg = data[r]['stg']; sit = data[r]['sit']
        n_total = len(stg)
        n_reached = int(np.isfinite(stg).sum())
        stg_clipped = np.where(np.isfinite(stg), stg, float(args.max_steps))
        stg_means.append(float(stg_clipped.mean()))
        stg_cis.append(float(ci95(stg_clipped)))
        sit_means.append(float(sit.mean()))
        sit_cis.append(float(ci95(sit)))
        rr = n_reached / n_total
        reach_rates.append(rr)
        print(f"SAC-R{r}: seeds={sorted(data[r]['seeds'])} n={n_total} "
              f"reached={n_reached} ({rr:.1%})  "
              f"steps_to_goal={stg_clipped.mean():.0f}±{ci95(stg_clipped):.0f}  "
              f"steps_in_target={sit.mean():.0f}±{ci95(sit):.0f}")

    seeds_used = sorted(set().union(*[data[r]['seeds'] for r in data]))
    x = np.arange(len(rewards))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(x, stg_means, yerr=stg_cis, capsize=6,
                color=['C0', 'C1', 'C2'])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"SAC-R{r}" for r in rewards])
    axes[0].set_ylabel('steps to goal (lower = better)')
    axes[0].set_title(f'Steps to reach target (cap={args.max_steps})')
    axes[0].grid(True, alpha=0.3, axis='y')
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

    fig.suptitle(f"Q2.3.3(a) — final-policy eval (seeds={seeds_used})",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
