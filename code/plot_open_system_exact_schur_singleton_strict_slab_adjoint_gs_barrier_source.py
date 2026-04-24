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
    prefix = 'chi_open_system_exact_schur_singleton_strict_slab_adjoint_gs_barrier_source'
    summary = pd.read_csv(OUT_DIR / f'{prefix}_summary.csv').iloc[0]
    edges = pd.read_csv(OUT_DIR / f'{prefix}_edges.csv')
    detail = pd.read_csv(OUT_DIR / f'{prefix}_detail.csv')
    iterations = pd.read_csv(OUT_DIR / f'{prefix}_iterations.csv')

    product_sweeps = int(summary['product_sweeps'])
    floor_sweeps = int(summary['floor_sweeps'])
    global_ratio = float(summary['product_sweep_direct_flux_product_over_global_after_source'])
    local_ratio = float(summary['product_sweep_outer_bottom_over_local_after_source'])

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8))

    ax = axes[0]
    ax.plot(
        iterations['sweep'],
        iterations['direct_flux_product_over_global_after_source'],
        marker='o',
        linewidth=1.1,
        label='global product ratio',
    )
    ax.plot(
        iterations['sweep'],
        iterations['outer_bottom_over_local_after_source'],
        marker='.',
        linewidth=1.1,
        label='outer+bottom local ratio',
    )
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax.axvline(product_sweeps, color='tab:green', linestyle=':', linewidth=1.2, label=f'{product_sweeps} sweeps')
    ax.axvline(floor_sweeps, color='tab:red', linestyle=':', linewidth=1.2, label=f'{floor_sweeps} sweeps')
    ax.set_xlabel('monotone GS sweep')
    ax.set_ylabel('certified ratio')
    ax.set_title('finite GS lower iterates close the trace')
    ax.legend(loc='best', fontsize=8)

    ax = axes[1]
    labels = edges['edge_label']
    x = range(len(edges))
    width = 0.24
    ax.bar([i - width for i in x], edges['gs_product_sweep_flux_product'], width=width, label=f'GS {product_sweeps}')
    ax.bar([i for i in x], edges['gs_floor_sweep_flux_product'], width=width, label=f'GS {floor_sweeps}')
    ax.bar([i + width for i in x], edges['old_4sig_floor_flux_product'], width=width, label='old 4-sig p floor')
    ax.set_xticks(list(x), labels)
    ax.set_ylabel('direct flux product')
    ax.set_title(f'GS {product_sweeps}: global={global_ratio:.6f}, local={local_ratio:.6f}')
    ax.legend(loc='best', fontsize=8)

    ax = axes[2]
    for edge_label, group in detail.groupby('edge_label', sort=False):
        xvals = group['z_rel'] if edge_label in {'out', 'rb', 'lb'} else group['rho']
        ax.plot(
            xvals,
            group['product_sweep_over_4sig_floor'],
            marker='o',
            linewidth=1.0,
            label=f'{edge_label} GS {product_sweeps}',
        )
        ax.plot(
            xvals,
            group['floor_sweep_over_4sig_floor'],
            marker='.',
            linestyle='--',
            linewidth=0.9,
            label=f'{edge_label} GS {floor_sweeps}',
        )
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_xlabel(r'$z-D/2$ for outer/corners, $\rho$ for bottom')
    ax.set_ylabel(r'$p^{(n)}_q/p_q^-$')
    ax.set_title('floor compatibility is optional but audited')
    ax.legend(loc='best', fontsize=6, ncols=2)

    for ax in axes:
        ax.grid(alpha=0.18, linewidth=0.6)

    fig.tight_layout()
    out = OUT_DIR / f'{prefix}_figure.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
