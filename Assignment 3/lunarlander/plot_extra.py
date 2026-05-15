"""Extra diagnostic plots generated from existing progress.csv data.

Generates:
  lander_alpha_continuous.png   — temperature α evolution over training (Q2.2.1)
  lander_alpha_hover.png        — α evolution for fixed vs auto, with swap line (Q2.2.3)
  lander_final_bar.png          — final-performance bar chart across all 5 variants
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = Path(__file__).parent / 'runs'
SEEDS     = list(range(1, 16))
SWAP_STEP = 250_000

STYLE = {
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 11,
}
plt.rcParams.update(STYLE)

PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
SEED_KW = dict(color='#bbbbbb', linewidth=0.55, alpha=0.4, zorder=1)


# ── helpers ───────────────────────────────────────────────────────────────────

def load(path: Path):
    steps, returns, alphas = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row['step']))
            returns.append(float(row['return_mean']))
            a = row.get('alpha', '')
            alphas.append(float(a) if a.strip() else float('nan'))
    return np.asarray(steps), np.asarray(returns), np.asarray(alphas)


def collect(prefix_list, seeds=None):
    """For each prefix, aggregate steps/returns/alphas over given seeds.
    Returns list of (steps, mean_ret, ci_ret, mean_alpha, ci_alpha, per_seed_ret)
    """
    if seeds is None:
        seeds = SEEDS
    out = []
    for prefix in prefix_list:
        all_ret, all_alpha = [], []
        steps_ref = None
        for s in seeds:
            p = RUNS / f'{prefix}_seed{s}' / 'progress.csv'
            if not p.exists():
                continue
            st, ret, alp = load(p)
            if steps_ref is None:
                steps_ref = st
            L = min(len(steps_ref), len(ret))
            all_ret.append(ret[:L])
            all_alpha.append(alp[:L])
        if not all_ret:
            out.append(None)
            continue
        n = len(all_ret)
        L = min(len(r) for r in all_ret)
        steps_ref = steps_ref[:L]
        arr_ret = np.stack([r[:L] for r in all_ret])
        arr_alp = np.stack([a[:L] for a in all_alpha])
        mean_ret   = arr_ret.mean(0)
        ci_ret     = 1.96 * arr_ret.std(0) / np.sqrt(n)
        mean_alpha = np.nanmean(arr_alp, axis=0)
        ci_alpha   = 1.96 * np.nanstd(arr_alp, axis=0) / np.sqrt(n)
        out.append((steps_ref, mean_ret, ci_ret, mean_alpha, ci_alpha, arr_ret))
    return out


# ── plot 1: alpha evolution — continuous SAC ──────────────────────────────────

def plot_alpha_continuous():
    out = 'lander_alpha_continuous.png'
    res = collect(['sac_auto'])
    if res[0] is None:
        print(f'SKIP {out} — no data'); return

    steps, _, _, mean_alpha, ci_alpha, _ = res[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: return
    res_full = collect(['sac_auto'])
    st, m_ret, ci_ret, *_ = res_full[0]
    for s in SEEDS:
        p = RUNS / f'sac_auto_seed{s}' / 'progress.csv'
        if p.exists():
            _, ret, _ = load(p)
            ax1.plot(st[:len(ret)], ret[:len(st)], **SEED_KW)
    ax1.plot(st, m_ret, color=PALETTE[0], linewidth=2, label='Mean return', zorder=3)
    ax1.fill_between(st, m_ret - ci_ret, m_ret + ci_ret,
                     color=PALETTE[0], alpha=0.25, zorder=2)
    ax1.set_ylabel('Avg Undiscounted Return', fontsize=11)
    ax1.set_title('LunarLander Continuous SAC — Return & Temperature α', fontsize=13)
    ax1.legend(fontsize=10)

    # Bottom: alpha
    ax2.plot(steps, mean_alpha, color='#d62728', linewidth=2,
             label='Mean α (auto-tuned)', zorder=3)
    ax2.fill_between(steps, mean_alpha - ci_alpha, mean_alpha + ci_alpha,
                     color='#d62728', alpha=0.25, zorder=2)
    ax2.set_xlabel('Environment Timesteps', fontsize=11)
    ax2.set_ylabel('Temperature α', fontsize=11)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')


# ── plot 2: alpha evolution — hover experiment ────────────────────────────────

def plot_alpha_hover():
    out = 'lander_alpha_hover.png'
    prefixes = ['sac_manual_a0.01_hov200_swap250000to-100',
                'sac_auto_hov200_swap250000to-100']
    labels   = ['Fixed α = 0.01', 'Auto α']
    results  = collect(prefixes)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for i, (res, label, col) in enumerate(zip(results, labels, PALETTE)):
        if res is None:
            continue
        steps, m_ret, ci_ret, m_alp, ci_alp, arr_ret = res
        for row in arr_ret:
            ax1.plot(steps, row, **SEED_KW)
        ax1.plot(steps, m_ret, color=col, linewidth=2, label=label, zorder=3)
        ax1.fill_between(steps, m_ret - ci_ret, m_ret + ci_ret,
                         color=col, alpha=0.22, zorder=2)

        ax2.plot(steps, m_alp, color=col, linewidth=2, label=label, zorder=3)
        ax2.fill_between(steps, m_alp - ci_alp, m_alp + ci_alp,
                         color=col, alpha=0.22, zorder=2)

    for ax in (ax1, ax2):
        ax.axvline(SWAP_STEP, color='black', linestyle='--', linewidth=1.5,
                   alpha=0.7, label='Reward swap' if ax is ax1 else '_nolegend_',
                   zorder=4)

    ax1.set_ylabel('Avg Undiscounted Return', fontsize=11)
    ax1.set_title('LunarLander Hover-box — Return & α Before/After Reward Swap', fontsize=13)
    ax1.legend(fontsize=10)
    ax2.set_ylabel('Temperature α', fontsize=11)
    ax2.set_xlabel('Environment Timesteps', fontsize=11)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')


# ── plot 3: final-performance bar chart ───────────────────────────────────────

def plot_final_bar():
    out = 'lander_final_bar.png'

    configs = [
        ('sac_auto',                             'Continuous\nSAC (auto α)'),
        ('sac_manual_a0.01_hov200_swap250000to-100', 'Hover\nFixed α=0.01'),
        ('sac_auto_hov200_swap250000to-100',         'Hover\nAuto α'),
        ('sac_discrete',                         'Discrete\nSAC'),
        ('dqn',                                  'DQN'),
    ]

    means_final, cis_final = [], []
    for prefix, _ in configs:
        finals = []
        for s in SEEDS:
            p = RUNS / f'{prefix}_seed{s}' / 'progress.csv'
            if not p.exists():
                continue
            _, ret, _ = load(p)
            finals.append(ret[-1])
        if finals:
            n = len(finals)
            arr = np.array(finals)
            means_final.append(arr.mean())
            cis_final.append(1.96 * arr.std() / np.sqrt(n))
        else:
            means_final.append(float('nan'))
            cis_final.append(0.0)

    labels = [c[1] for c in configs]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, means_final, yerr=cis_final,
                  color=PALETTE[:len(labels)],
                  capsize=6, width=0.55, zorder=3,
                  error_kw=dict(linewidth=1.5, capthick=1.5))

    # Annotate each bar with its value
    for bar, m, ci in zip(bars, means_final, cis_final):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + ci + 4,
                    f'{m:.1f}', ha='center', va='bottom', fontsize=10)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Average Undiscounted Return at 500K steps', fontsize=11)
    ax.set_title('LunarLander — Final Performance Comparison (mean ± 95% CI)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent)  # run from Assignment 3/
    plot_alpha_continuous()
    plot_alpha_hover()
    plot_final_bar()
