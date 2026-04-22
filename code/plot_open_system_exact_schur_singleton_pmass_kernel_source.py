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
    s = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_pmass_kernel_source_summary.csv').iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)

    ax = axes[0]
    labels = [r'$K_{\min}W_+$', r'$P_{\rm req,g}$', r'$P_{\rm req,l}$', r'$P_{\rm mass}$']
    vals = [
        float(s['lower_product_from_kernel_min']),
        float(s['required_Pmass_global']),
        float(s['required_Pmass_local']),
        float(s['P_mass_11']),
    ]
    colors = ['#8D99AE', '#457B9D', '#E63946', '#2A9D8F']
    ax.bar(np.arange(len(labels)), vals, color=colors, alpha=0.94)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=14)
    ax.set_ylabel('mass scale')
    ax.set_title(r'D=11 Coarse Kernel Floor vs Exact Mass Carrier')
    ax.text(
        0.02, 0.97,
        rf"$K_{{\min}}W_+/P_{{req,g}}$ = {float(s['lower_product_over_required_global']):.4f}" "\n"
        rf"$K_{{\min}}W_+/P_{{req,l}}$ = {float(s['lower_product_over_required_local']):.4f}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    ax = axes[1]
    labels = [r'$W_{{+,req,g}}$', r'$W_{{+,req,l}}$', r'$W_+$']
    vals = [
        float(s['required_Wpos_global_given_kavg']),
        float(s['required_Wpos_local_given_kavg']),
        float(s['W_pos_11']),
    ]
    ax.bar(np.arange(len(labels)), vals, color=['#457B9D', '#E63946', '#2A9D8F'], alpha=0.94)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=14)
    ax.set_ylabel('carrier moment')
    ax.set_title(r'D=11 Positive-Carrier Route')
    ax.text(
        0.02, 0.97,
        rf"$W_+/W_{{req,g}}$ = {float(s['Wpos_over_required_global']):.4f}" "\n"
        rf"$W_+/W_{{req,l}}$ = {float(s['Wpos_over_required_local']):.4f}" "\n"
        rf"$W_-/W_+$ = {float(s['carrier_neg_over_pos_11']):.4f}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    ax = axes[2]
    labels = [r'$\bar K_{{+,req,g}}$', r'$\bar K_{{+,req,l}}$', r'$\bar K_+$', r'$K_{\min}$']
    vals = [
        float(s['required_kavg_global_given_Wpos']),
        float(s['required_kavg_local_given_Wpos']),
        float(s['mass_kernel_avg_weighted']),
        float(s['mass_kernel_min_anchor']),
    ]
    ax.bar(np.arange(len(labels)), vals, color=['#457B9D', '#E63946', '#2A9D8F', '#8D99AE'], alpha=0.94)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=14)
    ax.set_ylabel('kernel scale')
    ax.set_title(r'D=11 Weighted-Mass-Kernel Route')
    ax.text(
        0.02, 0.97,
        rf"$\bar K_+/\bar K_{{req,g}}$ = {float(s['kavg_over_required_global']):.4f}" "\n"
        rf"$\bar K_+/\bar K_{{req,l}}$ = {float(s['kavg_over_required_local']):.4f}" "\n"
        rf"plus/minus $\bar K$ gap = {float(s['plus_minus_kavg_rel_gap']):.4f}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_pmass_kernel_source_figure.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
