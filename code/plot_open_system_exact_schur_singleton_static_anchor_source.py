#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    s = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_static_anchor_source_summary.csv').iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)

    ax = axes[0]
    labels = ['pair-core', 'anchor', 'shell', 'req core g', 'req core l']
    vals = [
        float(s['self_pair_core']),
        float(s['self_anchor']),
        float(s['self_shell']),
        float(s['required_core_floor_global']),
        float(s['required_core_floor_local']),
    ]
    colors = ['#264653', '#2A9D8F', '#E9C46A', '#457B9D', '#E63946']
    ax.bar(np.arange(len(labels)), vals, color=colors, alpha=0.94)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=16)
    ax.set_ylabel('magnitude')
    ax.set_title(r'D=11 Core to Anchor Reduction')
    ax.text(
        0.02,
        0.97,
        rf"anchor/core = {float(s['anchor_abs_share_of_core']):.4f}" "\n"
        rf"shell/core = {float(s['shell_abs_share_of_core']):.4f}",
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    ax = axes[1]
    labels = ['diag anchor', 'mirror loss', 'induced anchor', 'req diag g', 'req diag l']
    vals = [
        float(s['diag_anchor']),
        float(s['mirror_anchor']),
        float(s['induced_anchor_from_diag']),
        float(s['required_diag_floor_global']),
        float(s['required_diag_floor_local']),
    ]
    colors = ['#2A9D8F', '#E76F51', '#264653', '#457B9D', '#E63946']
    ax.bar(np.arange(len(labels)), vals, color=colors, alpha=0.94)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=16)
    ax.set_ylabel('magnitude')
    ax.set_title(r'D=11 Static Anchor Bottleneck')
    ax.text(
        0.02,
        0.97,
        rf"$\mu_{{11}}^{{anc}}$ = {float(s['mu_anchor_11']):.4f}" "\n"
        rf"diag/req$_g$ = {float(s['diag_over_required_global']):.4f}" "\n"
        rf"diag/req$_l$ = {float(s['diag_over_required_local']):.4f}",
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_static_anchor_source_figure.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
