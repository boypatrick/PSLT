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
        / 'chi_open_system_exact_schur_singleton_strict_slab_edge_certificate_source_summary.csv'
    ).iloc[0]
    detail = pd.read_csv(
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_edge_certificate_source_detail.csv'
    )

    target_global = float(summary['target_global_after_source'])
    target_local = float(summary['target_local_after_source'])
    gap = float(summary['certified_gap_over_global_after_source'])
    labels = detail['edge_label']
    x = range(len(detail))

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    width = 0.36
    ax.bar(
        [i - width / 2 for i in x],
        detail['exact_product'],
        width=width,
        label='exact edge product',
        color='tab:blue',
    )
    ax.bar(
        [i + width / 2 for i in x],
        detail['certified_product_floor'],
        width=width,
        label='rounded certified floor',
        color='tab:orange',
    )
    ax.set_xticks(list(x), labels)
    ax.set_ylabel('edge product mass')
    ax.set_title(r'exact products vs certified lower floors')
    ax.legend(loc='best', fontsize=9)

    ax = axes[1]
    ax.plot(
        labels,
        detail['cumulative_certified_product'],
        marker='o',
        color='tab:purple',
        label='certified cumulative floor',
    )
    ax.axhline(target_global, color='black', linestyle='--', linewidth=1.0, label='global trace target')
    ax.axhline(target_local, color='tab:green', linestyle=':', linewidth=1.2, label='local trace target')
    ax.set_ylabel('cumulative certified mass')
    ax.set_title(r'rounded certificate closes the physical trace')
    ax.text(2.1, target_global + gap * 0.22, f'gap={gap:.2e}', fontsize=9)
    ax.legend(loc='best', fontsize=9)

    ax = axes[2]
    ax.plot(labels, detail['boundary_defect_over_floor'], marker='o', label=r'$B_\sigma/B^-_\sigma$')
    ax.plot(labels, detail['average_weight_over_floor'], marker='s', label=r'$\bar p_\sigma/p^-_\sigma$')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_ylabel('exact / certified floor')
    ax.set_title(r'each scalar lower bound has positive slack')
    ax.legend(loc='best', fontsize=9)

    fig.tight_layout()
    out = (
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_edge_certificate_source_figure.png'
    )
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
