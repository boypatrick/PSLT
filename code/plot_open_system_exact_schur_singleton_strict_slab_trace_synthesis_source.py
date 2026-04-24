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
    prefix = 'chi_open_system_exact_schur_singleton_strict_slab_trace_synthesis_source'
    summary = pd.read_csv(OUT_DIR / f'{prefix}_summary.csv').iloc[0]
    stages = pd.read_csv(OUT_DIR / f'{prefix}_stages.csv')
    edges = pd.read_csv(OUT_DIR / f'{prefix}_edges.csv')

    target = float(summary['target_global_after_source'])
    local_target = float(summary['target_local_after_source'])
    required_global = float(summary['required_global'])
    source = float(summary['source_potential_mass'])

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 4.9))

    ax = axes[0]
    labels = ['O66', 'O68', 'O69', 'O70/548', 'O70/859']
    colors = ['#8f8f8f', '#4e79a7', '#59a14f', '#f28e2b', '#e15759']
    ax.bar(labels, stages['trace_product'], color=colors)
    ax.axhline(target, color='black', linestyle='--', linewidth=1.0, label='after-source global target')
    ax.set_ylabel(r'certified $I_{\partial,\sharp}^{phys}$')
    ax.set_title('closed trace certificate stack')
    ax.tick_params(axis='x', rotation=20)
    ax.legend(loc='best', fontsize=8)

    ax = axes[1]
    ax.plot(labels, stages['source_completed_over_required_global'], marker='o', linewidth=1.2)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_ylabel('source-completed global ratio')
    ax.set_title(r'$I_{\partial,\sharp}^{phys}+I_{src,\sharp}$ stays above target')
    ax.tick_params(axis='x', rotation=20)

    ax = axes[2]
    edge_labels = edges['edge_label']
    x = range(len(edges))
    width = 0.28
    ax.bar([i - width for i in x], edges['gs548_product'], width=width, label='GS548')
    ax.bar([i for i in x], edges['gs859_product'], width=width, label='GS859')
    ax.bar([i + width for i in x], edges['old_4sig_product'], width=width, label='old p-floor')
    ax.axhline(local_target, color='tab:green', linestyle=':', linewidth=1.0, label='local target')
    ax.set_xticks(list(x), edge_labels)
    ax.set_ylabel('edge contribution')
    ax.set_title('outer+bottom closes the local target')
    ax.legend(loc='best', fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.16, linewidth=0.6)

    fig.suptitle(
        f'parent-side closed; main ratio={float(summary["main_source_completed_over_required_global"]):.6f}, '
        f'target={required_global:.6e}, source={source:.6e}',
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT_DIR / f'{prefix}_figure.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
