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
    prefix = 'chi_open_system_exact_schur_singleton_strict_slab_gs_rate_source'
    summary = pd.read_csv(OUT_DIR / f'{prefix}_summary.csv').iloc[0]
    checkpoints = pd.read_csv(OUT_DIR / f'{prefix}_checkpoints.csv')

    product_sweeps = int(summary['product_sweeps'])
    floor_sweeps = int(summary['floor_sweeps'])
    first_global = int(summary['first_global_sweep'])
    first_local = int(summary['first_local_sweep'])
    first_floor = int(summary['first_point_floor_sweep'])

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8))

    ax = axes[0]
    ax.plot(
        checkpoints['sweep'],
        checkpoints['flux_product_over_global_after_source'],
        marker='o',
        label='global trace ratio',
    )
    ax.plot(
        checkpoints['sweep'],
        checkpoints['outer_bottom_over_local_after_source'],
        marker='s',
        label='outer+bottom local ratio',
    )
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax.axvline(first_global, color='tab:blue', linestyle=':', label=f'first global {first_global}')
    ax.axvline(first_local, color='tab:green', linestyle=':', label=f'first both {product_sweeps}')
    ax.axvline(first_floor, color='tab:red', linestyle=':', label=f'point floor {floor_sweeps}')
    ax.set_xlabel('GS sweep')
    ax.set_ylabel('certified ratio')
    ax.set_title('closure sweeps are audited, not tuned')
    ax.legend(loc='best', fontsize=8)

    ax = axes[1]
    ax.semilogy(
        checkpoints['sweep'],
        checkpoints['flux_product_tail_bound'],
        marker='o',
        label='global product tail bound',
    )
    ax.semilogy(
        checkpoints['sweep'],
        checkpoints['exact_flux_product_gap'],
        marker='.',
        linestyle='--',
        label='exact product gap',
    )
    ax.semilogy(
        checkpoints['sweep'],
        checkpoints['outer_bottom_tail_bound'],
        marker='s',
        label='local tail bound',
    )
    ax.semilogy(
        checkpoints['sweep'],
        checkpoints['exact_outer_bottom_gap'],
        marker='.',
        linestyle='--',
        label='exact local gap',
    )
    ax.set_xlabel('GS sweep')
    ax.set_ylabel('tail size')
    ax.set_title(r'$d_{n+1}\leq \alpha_n d_n$ tail certificates')
    ax.legend(loc='best', fontsize=8)

    ax = axes[2]
    ax.plot(
        checkpoints['sweep'],
        checkpoints['alpha_component_upper'],
        marker='o',
        label=r'$\alpha_n=\max_i d_{n+1,i}/d_{n,i}$',
    )
    ax.plot(
        checkpoints['sweep'],
        checkpoints['beta_component_lower'],
        marker='.',
        linestyle='--',
        label=r'componentwise min ratio',
    )
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_xlabel('GS sweep')
    ax.set_ylabel('increment ratio')
    ax.set_title('componentwise subeigenvector rate')
    ax.legend(loc='best', fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.18, linewidth=0.6)

    fig.tight_layout()
    out = OUT_DIR / f'{prefix}_figure.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
