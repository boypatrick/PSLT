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
    summary = pd.read_csv(
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_boundary_flux_certificate_source_summary.csv'
    ).iloc[0]
    edges = pd.read_csv(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_boundary_flux_certificate_source_edges.csv'
    )
    detail = pd.read_csv(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_boundary_flux_certificate_source_detail.csv'
    )

    target = float(summary['target_global_after_source'])
    local_target = float(summary['target_local_after_source'])
    gap = float(summary['certified_edge_gap_over_global_after_source'])

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    for edge_label, group in detail.groupby('edge_label', sort=False):
        x = group['z_rel'] if edge_label in {'out', 'rb', 'lb'} else group['rho']
        ax.plot(x, group['exact_boundary_defect'], marker='o', linewidth=1.2, label=f'{edge_label} exact')
        ax.plot(
            x,
            group['certified_flux_floor'],
            marker='.',
            linestyle='--',
            linewidth=1.0,
            label=f'{edge_label} flux floor',
        )
        ax.plot(
            x,
            group['certified_boundary_defect_floor'],
            marker='x',
            linestyle=':',
            linewidth=1.0,
            label=f'{edge_label} b-floor',
        )
    ax.set_xlabel(r'$z-D/2$ for outer/corners, $\rho$ for bottom')
    ax.set_ylabel(r'boundary defect $b_q$')
    ax.set_title(r'missing-neighbor flux certifies $b_q^-$')
    ax.legend(loc='best', fontsize=6, ncols=2)

    ax = axes[1]
    labels = edges['edge_label']
    x = range(len(edges))
    width = 0.26
    ax.bar([i - width for i in x], edges['exact_product'], width=width, label='exact')
    ax.bar(
        [i for i in x],
        edges['flux_certified_product_with_p_floor'],
        width=width,
        label='flux with p-floor',
    )
    ax.bar([i + width for i in x], edges['certified_product_floor'], width=width, label='b/p floors')
    ax.set_xticks(list(x), labels)
    ax.set_ylabel('edge product')
    ax.set_title(r'flux-certified products retain the O66 margin')
    ax.legend(loc='best', fontsize=8)

    ax = axes[2]
    ax.plot(labels, edges['cumulative_certified_product'], marker='o', color='tab:purple')
    ax.axhline(target, color='black', linestyle='--', linewidth=1.0, label='global trace target')
    ax.axhline(local_target, color='tab:green', linestyle=':', linewidth=1.2, label='local trace target')
    ax.text(2.0, target + 0.18 * gap, f'gap={gap:.2e}', fontsize=9)
    ax.set_ylabel('cumulative certified product')
    ax.set_title(r'boundary-flux certificate still closes')
    ax.legend(loc='best', fontsize=9)

    fig.tight_layout()
    out = (
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_boundary_flux_certificate_source_figure.png'
    )
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
