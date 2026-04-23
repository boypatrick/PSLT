#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from audit_open_system_parity_contrast_self_mirror_source import solve_fields
from extract_chi_localized_2d import Level, PhysicalParams

OUT_DIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    s = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_carrier_source_summary.csv').iloc[0]

    D = float(s['D'])
    rho_anchor = float(s['rho_anchor'])
    zeta_anchor = float(s['zeta_anchor'])

    p = PhysicalParams()
    level = Level('fine', dr=0.06, dz=0.03)
    f = solve_fields(D=D, p=p, level=level, rho_max=3.0, z_margin=6.0, sigma=2.5, tol=1e-8, maxiter=30000)
    rr = f['rr']
    zz = f['zz']
    z = f['z']
    carrier_pos = np.clip(f['contrast_density'], 0.0, None)

    plus = (rr <= rho_anchor) & (np.abs(zz - D / 2.0) <= zeta_anchor)
    minus = (rr <= rho_anchor) & (np.abs(zz + D / 2.0) <= zeta_anchor)

    zeta_plus = z[(np.abs(z - D / 2.0) <= zeta_anchor)] - D / 2.0
    zeta_minus = -(z[(np.abs(z + D / 2.0) <= zeta_anchor)] + D / 2.0)
    plus_profile = carrier_pos[:, (np.abs(z - D / 2.0) <= zeta_anchor)][rr[:, 0] <= rho_anchor, :].sum(axis=0)
    minus_profile = carrier_pos[:, (np.abs(z + D / 2.0) <= zeta_anchor)][rr[:, 0] <= rho_anchor, :].sum(axis=0)[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), constrained_layout=True)

    ax = axes[0]
    labels = [r'$W_{+,\mathrm{req},g}/2$', r'$W_{+,\mathrm{req},l}/2$', r'$W_{11,+}^{(+)}$', r'$W_{11,-}^{(+)}$', r'$W_{11}^{(+)}/2$']
    vals = [
        float(s['required_half_global']),
        float(s['required_half_local']),
        float(s['W_pos_plus_11']),
        float(s['W_pos_minus_11']),
        float(s['half_total_W_pos_11']),
    ]
    colors = ['#457B9D', '#E63946', '#2A9D8F', '#8D99AE', '#264653']
    ax.bar(np.arange(len(labels)), vals, color=colors, alpha=0.95)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=16)
    ax.set_ylabel('single-lobe carrier moment')
    ax.set_title(r'D=11 Single-Lobe Thresholds')
    ax.text(
        0.02, 0.97,
        rf"$W_{{+,11}}/W_{{req,g}}^{{(1/2)}}$ = {float(s['plus_over_required_half_global']):.4f}" "\n"
        rf"$W_{{+,11}}/W_{{req,l}}^{{(1/2)}}$ = {float(s['plus_over_required_half_local']):.4f}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    ax = axes[1]
    labels = [r'$W_{11,+}^{(+)}$', r'$W_{11}^{(+)}/2$', r'$W_{11,-}^{(+)}$']
    vals = [
        float(s['W_pos_plus_11']),
        float(s['half_total_W_pos_11']),
        float(s['W_pos_minus_11']),
    ]
    ax.bar(np.arange(len(labels)), vals, color=['#2A9D8F', '#264653', '#8D99AE'], alpha=0.95)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=16)
    ax.set_ylabel('carrier moment')
    ax.set_title(r'D=11 Reflection-to-Half-Total Check')
    ax.text(
        0.02, 0.97,
        rf"sum residual = {float(s['lobe_sum_reconstruction_residual_abs']):.1e}" "\n"
        rf"plus-half gap = {float(s['plus_half_residual_abs']):.2e}" "\n"
        rf"minus-half gap = {float(s['minus_half_residual_abs']):.2e}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    ax = axes[2]
    ax.plot(zeta_plus, plus_profile, color='#2A9D8F', lw=2.2, label=r'plus lobe')
    ax.plot(zeta_minus, minus_profile, color='#8D99AE', lw=2.2, ls='--', label=r'mirrored minus lobe')
    ax.fill_between(zeta_plus, plus_profile, color='#2A9D8F', alpha=0.18)
    ax.fill_between(zeta_minus, minus_profile, color='#8D99AE', alpha=0.16)
    ax.axvline(0.0, color='0.35', lw=1.0, alpha=0.8)
    ax.set_xlabel(r'local axial coordinate $\zeta$')
    ax.set_ylabel('slice-integrated positive carrier')
    ax.set_title(r'D=11 Single-Lobe Carrier Profiles')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.text(
        0.02, 0.97,
        rf"plus/half-total = {float(s['plus_over_half_total']):.4f}" "\n"
        rf"minus/half-total = {float(s['minus_over_half_total']):.4f}" "\n"
        rf"lobe gap / total = {float(s['plus_minus_rel_total_gap']):.4f}",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9, edgecolor='0.8'),
    )

    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_carrier_source_figure.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
