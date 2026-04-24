#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / 'pslt_mpl_cache'
(_CACHE_ROOT / 'mpl').mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / 'xdg').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_CACHE_ROOT / 'mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(_CACHE_ROOT / 'xdg'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    prefix = 'chi_open_system_exact_schur_singleton_strict_slab_exterior_phi_barrier_source'
    summary = pd.read_csv(OUT_DIR / f'{prefix}_summary.csv').iloc[0]
    edges = pd.read_csv(OUT_DIR / f'{prefix}_edges.csv')
    trace = pd.read_csv(OUT_DIR / f'{prefix}_trace.csv')
    detail = pd.read_csv(OUT_DIR / f'{prefix}_detail.csv')

    target = float(summary['target_global_after_source'])
    local_target = float(summary['target_local_after_source'])
    theta = float(summary['phi_barrier_theta'])
    gap = float(summary['direct_flux_gap_over_global_after_source'])

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    x = range(len(trace))
    ax.plot(x, trace['phi_exact'], marker='o', linewidth=1.1, label=r'$\phi_L$ exact')
    ax.plot(x, trace['phi_barrier'], marker='.', linestyle='--', linewidth=1.0, label=rf'$\theta\phi_L$, $\theta={theta}$')
    ax.plot(x, trace['phi_floor'], marker='x', linestyle=':', linewidth=1.0, label='4-sig floor')
    ax.set_xlabel('exterior trace point')
    ax.set_ylabel(r'$\phi_L$')
    ax.set_title(r'parent trace barrier dominates 4-sig floors')
    ax.legend(loc='best', fontsize=8)

    ax = axes[1]
    labels = edges['edge_label']
    xloc = range(len(edges))
    width = 0.26
    ax.bar(
        [i - width for i in xloc],
        edges['old_certified_product_floor'],
        width=width,
        label='O66 b/p floors',
    )
    ax.bar(
        [i for i in xloc],
        edges['barrier_flux_product_with_p_floor'],
        width=width,
        label='barrier flux',
    )
    ax.bar(
        [i + width for i in xloc],
        edges['direct_flux_product_with_p_floor'],
        width=width,
        label='4-sig flux',
    )
    ax.set_xticks(list(xloc), labels)
    ax.set_ylabel('edge product')
    ax.set_title(r'direct flux route keeps the trace margin')
    ax.legend(loc='best', fontsize=8)

    ax = axes[2]
    ax.plot(labels, edges['cumulative_direct_flux_product'], marker='o', color='tab:blue', label='4-sig flux product')
    ax.plot(labels, edges['cumulative_barrier_product'], marker='.', color='tab:orange', label='barrier product')
    ax.axhline(target, color='black', linestyle='--', linewidth=1.0, label='global trace target')
    ax.axhline(local_target, color='tab:green', linestyle=':', linewidth=1.2, label='local trace target')
    ax.text(2.0, target + 0.18 * gap, f'gap={gap:.2e}', fontsize=9)
    ax.set_ylabel('cumulative certified product')
    ax.set_title(r'outer+bottom still closes local target')
    ax.legend(loc='best', fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.18, linewidth=0.6)

    fig.tight_layout()
    out = OUT_DIR / f'{prefix}_figure.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
