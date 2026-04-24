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
    s = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_pp_factor_source_summary.csv').iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), constrained_layout=True)

    ax = axes[0]
    labels = [r'$W_{++}$', r'$S_{L,+}^{(+)}\bar\Phi_R^{(+|L+)}$', r'$S_{R,+}^{(+)}\bar\Phi_L^{(+|R+)}$', r'$W_{\rm req,g}^{(1/2)}$', r'$W_{\rm req,l}^{(1/2)}$']
    vals = [
        float(s['single_lobe_positive_positive_overlap']),
        float(s['single_lobe_mirror_positive_mass']) * float(s['weighted_same_center_avg_given_mirror_positive']),
        float(s['single_lobe_same_center_positive_mass']) * float(s['weighted_mirror_positive_avg_given_same_center']),
        float(s['required_half_global']),
        float(s['required_half_local']),
    ]
    ax.bar(np.arange(len(labels)), vals, color=['#2A9D8F', '#457B9D', '#8D99AE', '#E76F51', '#C1121F'], alpha=0.95)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=16)
    ax.set_ylabel('single-lobe overlap scale')
    ax.set_title(r'$D=11$: exact single-lobe $++$ factorizations')
    ax.text(
        0.02, 0.97,
        rf"mirror-route resid = {float(s['mirror_route_factorization_residual_abs']):.1e}" "\n"
        rf"self-route resid = {float(s['same_center_route_factorization_residual_abs']):.1e}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    ax = axes[1]
    labels = [r'$S_{L,+}^{(+)}$', r'$S_{L,+,\rm req}^{(g)}$', r'$S_{L,+,\rm req}^{(l)}$', r'$S_{R,+}^{(+)}$', r'$S_{R,+,\rm req}^{(g)}$', r'$S_{R,+,\rm req}^{(l)}$']
    vals = [
        float(s['single_lobe_mirror_positive_mass']),
        float(s['required_mirror_positive_mass_global']),
        float(s['required_mirror_positive_mass_local']),
        float(s['single_lobe_same_center_positive_mass']),
        float(s['required_same_center_positive_mass_global']),
        float(s['required_same_center_positive_mass_local']),
    ]
    ax.bar(np.arange(len(labels)), vals, color=['#457B9D', '#E76F51', '#C1121F', '#2A9D8F', '#F4A261', '#E63946'], alpha=0.95)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=18)
    ax.set_ylabel('single-function positive mass')
    ax.set_title(r'$D=11$: mass routes for $W_{11,+}^{(++)}$')
    ax.text(
        0.02, 0.97,
        rf"$S_{{L,+}}/S_{{L,+,req}}^{{(g)}}$ = {float(s['mirror_mass_over_required_global']):.4f}" "\n"
        rf"$S_{{R,+}}/S_{{R,+,req}}^{{(g)}}$ = {float(s['same_center_mass_over_required_global']):.4f}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    ax = axes[2]
    labels = [r'$\bar\Phi_R^{(+|L+)}$', r'$\bar\Phi_{R,\rm req}^{(g)}$', r'$\bar\Phi_{R,\rm req}^{(l)}$', r'$\bar\Phi_L^{(+|R+)}$', r'$\bar\Phi_{L,\rm req}^{(g)}$', r'$\bar\Phi_{L,\rm req}^{(l)}$']
    vals = [
        float(s['weighted_same_center_avg_given_mirror_positive']),
        float(s['required_weighted_same_center_avg_global']),
        float(s['required_weighted_same_center_avg_local']),
        float(s['weighted_mirror_positive_avg_given_same_center']),
        float(s['required_weighted_mirror_avg_global']),
        float(s['required_weighted_mirror_avg_local']),
    ]
    ax.bar(np.arange(len(labels)), vals, color=['#2A9D8F', '#F4A261', '#E76F51', '#457B9D', '#8D99AE', '#C1121F'], alpha=0.95)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=18)
    ax.set_ylabel('weighted positive average')
    ax.set_title(r'$D=11$: multiplier routes and preferred theorem line')
    ax.text(
        0.02, 0.97,
        rf"$\phi_R>0$ on $\{{\phi_L>0\}}$: {int(s['phiR_positive_on_phiL_positive_support'])}" "\n"
        rf"min$\,\phi_R|_{{\phi_L>0}}$ = {float(s['min_phiR_on_phiL_positive_support']):.3e}" "\n"
        r"preferred route: $S_{L,+}^{(+)}$",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_pp_factor_source_figure.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
