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
    s = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_overlap_source_summary.csv').iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)

    ax = axes[0]
    labels = [r'$W_{+,\mathrm{req},g}^{(1/2)}$', r'$W_{+,\mathrm{req},l}^{(1/2)}$', r'$W_{++}$', r'$W_{--}$', r'$W_{+,\mathrm{exact}}$']
    vals = [
        float(s['required_half_global']),
        float(s['required_half_local']),
        float(s['single_lobe_positive_positive_overlap']),
        float(s['single_lobe_negative_negative_completion']),
        float(s['single_lobe_positive_overlap_exact']),
    ]
    colors = ['#457B9D', '#E63946', '#2A9D8F', '#8D99AE', '#264653']
    ax.bar(np.arange(len(labels)), vals, color=colors, alpha=0.95)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=16)
    ax.set_ylabel('single-lobe overlap scale')
    ax.set_title(r'D=11 Single-Lobe Positive-Overlap Route')
    ax.text(
        0.02, 0.97,
        rf"$W_{{++}}/W_{{req,g}}^{{(1/2)}}$ = {float(s['pp_over_required_half_global']):.4f}" "\n"
        rf"$W_{{++}}/W_{{req,l}}^{{(1/2)}}$ = {float(s['pp_over_required_half_local']):.4f}" "\n"
        rf"$W_{{--}}/W_{{+,exact}}$ = {float(s['mm_over_exact']):.4e}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    ax = axes[1]
    labels = [r'$A_{++}$', r'$B_{+-}$', r'$C_{-+}$', r'$D_{--}$']
    vals = [
        float(s['A_pp']),
        float(s['B_pm']),
        float(s['C_mp']),
        float(s['D_mm']),
    ]
    ax.bar(np.arange(len(labels)), vals, color=['#2A9D8F', '#E76F51', '#CDB4DB', '#8D99AE'], alpha=0.95)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel('quadrant overlap integral')
    ax.set_title(r'D=11 Localized-State Sign Quadrants on $B_+(11)$')
    ax.text(
        0.02, 0.97,
        rf"$C_{{-+}}=0$ on grid: {int(s['C_mp_is_zero'])}" "\n"
        rf"$2(A+D)=W_{{+,exact}}$ residual = {float(s['positive_overlap_split_residual_abs']):.1e}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    ax = axes[2]
    labels = [r'$W_{++}$', r'$W_{+,exact}$', r'$\|\phi_{R,-}\|_1/\|\phi_{R,+}\|_1$']
    vals = [
        float(s['single_lobe_positive_positive_overlap']),
        float(s['single_lobe_positive_overlap_exact']),
        float(s['phiR_neg_over_pos_L1']),
    ]
    colors = ['#2A9D8F', '#264653', '#8D99AE']
    ax.bar(np.arange(2), vals[:2], color=colors[:2], alpha=0.95)
    ax2 = ax.twinx()
    ax2.bar([2], [vals[2]], color=colors[2], alpha=0.75)
    ax.set_xticks([0,1,2], [r'$W_{++}$', r'$W_{+,exact}$', r'$\|\phi_{R,-}\|_1/\|\phi_{R,+}\|_1$'])
    ax.set_ylabel('carrier scale')
    ax2.set_ylabel('same-center negative share')
    ax.set_title(r'D=11 Same-Center Sign Purity on the Plus Lobe')
    ax.text(
        0.02, 0.97,
        rf"$W_{{++}}/W_{{+,exact}}$ = {float(s['pp_over_exact']):.4f}" "\n"
        rf"$\|\phi_{{R,-}}\|_1/\|\phi_{{R,+}}\|_1$ = {float(s['phiR_neg_over_pos_L1']):.2e}" "\n"
        rf"neg-point frac$(\phi_R)$ = {float(s['phiR_neg_point_fraction']):.4f}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_overlap_source_figure.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
