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
    s = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_diag_mass_bridge_source_summary.csv').iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)

    ax = axes[0]
    labels = ['P_mass', 'N_der', 'Pdiag lb', 'P_diag req g', 'P_diag req l']
    vals = [
        float(s['P_mass_11']),
        float(s['N_der_11']),
        float(s['induced_Pdiag_lower_from_mass_minus_nder']),
        float(s['required_Pdiag_global']),
        float(s['required_Pdiag_local']),
    ]
    colors = ['#2A9D8F', '#E76F51', '#264653', '#457B9D', '#E63946']
    ax.bar(np.arange(len(labels)), vals, color=colors, alpha=0.94)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=16)
    ax.set_ylabel('magnitude')
    ax.set_title(r'D=11 Positive-Moment Lower Bound')
    ax.text(
        0.02, 0.97,
        rf"$N_{{der}}/P_{{mass}}$ = {float(s['nder_over_pmass']):.4f}" "\n"
        rf"lb/req$_g$ = {float(s['induced_over_required_global']):.4f}" "\n"
        rf"lb/req$_l$ = {float(s['induced_over_required_local']):.4f}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    ax = axes[1]
    labels = ['P req g', 'P req l', 'P actual', 'N max g', 'N max l', 'N actual']
    vals = [
        float(s['required_Pmass_global_given_Nder']),
        float(s['required_Pmass_local_given_Nder']),
        float(s['P_mass_11']),
        float(s['allowed_Nder_global_given_Pmass']),
        float(s['allowed_Nder_local_given_Pmass']),
        float(s['N_der_11']),
    ]
    colors = ['#457B9D', '#E63946', '#2A9D8F', '#457B9D', '#E63946', '#264653']
    ax.bar(np.arange(len(labels)), vals, color=colors, alpha=0.94)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=16)
    ax.set_ylabel('threshold scale')
    ax.set_title(r'D=11 Two Equivalent Mass/Derivative Routes')
    ax.text(
        0.02, 0.97,
        rf"P/req$_g$ = {float(s['Pmass_over_required_global']):.4f}" "\n"
        rf"P/req$_l$ = {float(s['Pmass_over_required_local']):.4f}" "\n"
        rf"$N_{{der}}$/N$_g^{{max}}$ = {float(s['Nder_over_allowed_global']):.4f}" "\n"
        rf"$N_{{der}}$/N$_l^{{max}}$ = {float(s['Nder_over_allowed_local']):.4f}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_diag_mass_bridge_source_figure.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
