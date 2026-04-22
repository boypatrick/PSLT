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
    s = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_diag_moment_source_summary.csv').iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)

    ax = axes[0]
    labels = ['P_diag', 'N_diag', 'I_diag', 'req I_g', 'req I_l']
    vals = [
        float(s['P_diag_11']),
        float(s['N_diag_11']),
        float(s['I_diag_anchor_11']),
        float(s['required_diag_floor_global']),
        float(s['required_diag_floor_local']),
    ]
    colors = ['#2A9D8F', '#E76F51', '#264653', '#457B9D', '#E63946']
    ax.bar(np.arange(len(labels)), vals, color=colors, alpha=0.94)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=15)
    ax.set_ylabel('magnitude')
    ax.set_title(r'D=11 Oriented-Moment Split')
    ax.text(
        0.02, 0.97,
        rf"$\nu_{{11}}$ = {float(s['nu_11']):.4f}" "\n"
        rf"$I=(1-\nu)P$ residual = {float(s['split_reconstruction_residual']):.1e}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    ax = axes[1]
    labels = ['P req g', 'P req l', 'P actual', r'$\nu$ max g', r'$\nu$ max l', r'$\nu$ actual']
    vals = [
        float(s['required_P_diag_global_given_nu11']),
        float(s['required_P_diag_local_given_nu11']),
        float(s['P_diag_11']),
        float(s['required_nu11_global_given_Pdiag']),
        float(s['required_nu11_local_given_Pdiag']),
        float(s['nu_11']),
    ]
    ax.bar(np.arange(3), vals[:3], color=['#457B9D','#E63946','#2A9D8F'], alpha=0.94)
    ax.bar(np.arange(3,6), vals[3:], color=['#457B9D','#E63946','#264653'], alpha=0.94)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=15)
    ax.set_ylabel('threshold scale')
    ax.set_title(r'D=11 Two Equivalent Parent-Side Routes')
    ax.text(
        0.02, 0.97,
        rf"P/req$_g$ = {float(s['Pdiag_over_required_global']):.4f}" "\n"
        rf"P/req$_l$ = {float(s['Pdiag_over_required_local']):.4f}" "\n"
        rf"$\Delta\nu_g$ = {float(s['nu_slack_global']):.4f}" "\n"
        rf"$\Delta\nu_l$ = {float(s['nu_slack_local']):.4f}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_diag_moment_source_figure.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
