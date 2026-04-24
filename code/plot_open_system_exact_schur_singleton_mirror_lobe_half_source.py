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
    s = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_lobe_half_source_summary.csv').iloc[0]
    rows = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_lobe_half_source_rows.csv')

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), constrained_layout=True)

    ax = axes[0]
    labels = [r'$S_{L,+}^{(+)}$', r'$I_{L,+}^{(\downarrow)}$', r'$N_{L,+}^{(\downarrow)}$', r'$S_{L,+,\rm req}^{(g)}$', r'$S_{L,+,\rm req}^{(l)}$']
    vals = [
        float(s['mirror_positive_lobe_mass']),
        float(s['mirror_signed_lower_half_mass']),
        float(s['mirror_negative_leakage_lower_half']),
        float(s['required_global']),
        float(s['required_local']),
    ]
    ax.bar(np.arange(len(labels)), vals, color=['#457B9D', '#2A9D8F', '#8D99AE', '#E76F51', '#C1121F'], alpha=0.95)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=16)
    ax.set_ylabel('mirror-positive lobe scale')
    ax.set_title(r'$D=11$: lower-half signed bridge for $S_{11,+}^{(L+)}$')
    ax.text(
        0.02, 0.97,
        rf"$I_{{L,+}}^{{(\downarrow)}}/S_{{L,+,req}}^{{(g)}}$ = {float(s['signed_lower_over_required_global']):.4f}" "\n"
        rf"$N_{{L,+}}^{{(\downarrow)}}/S_{{L,+}}^{{(+)}}$ = {float(s['negative_leakage_over_positive_mass']):.4e}" "\n"
        rf"bridge resid = {float(s['lower_half_bridge_residual_abs']):.1e}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    ax = axes[1]
    z = rows['z_rel'].to_numpy()
    ax.plot(z, rows['positive_mass'].to_numpy(), color='#2A9D8F', lw=2.0, label=r'positive row mass')
    ax.plot(z, rows['negative_mass'].to_numpy(), color='#8D99AE', lw=1.8, label=r'negative row mass')
    ax.plot(z, rows['signed_mass'].to_numpy(), color='#264653', lw=1.8, label=r'signed row mass')
    ax.axvline(0.0, color='k', ls='--', lw=1.0)
    ax.set_xlabel(r'$z-D/2$ on $B_+(11)$')
    ax.set_ylabel('row contribution')
    ax.set_title(r'$D=11$: where the mirror-positive mass lives')
    ax.legend(loc='upper left', fontsize=8)

    ax = axes[2]
    ax.plot(z, rows['positive_point_fraction'].to_numpy(), color='#457B9D', lw=2.0)
    ax.axvline(0.0, color='k', ls='--', lw=1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r'$z-D/2$ on $B_+(11)$')
    ax.set_ylabel(r'fraction of points with $\phi_L>0$')
    ax.set_title(r'$D=11$: sign geometry of $\phi_L$ on the plus lobe')
    ax.text(
        0.02, 0.97,
        rf"upper half nonpositive: {int(s['upper_half_is_nonpositive'])}" "\n"
        rf"lower positive-point frac = {float(s['lower_positive_point_fraction']):.4f}" "\n"
        rf"min lower / max upper = {float(s['min_phiL_lower']):.3e} / {float(s['max_phiL_upper']):.3e}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_lobe_half_source_figure.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
