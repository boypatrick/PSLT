#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'chi_open_system'
DETAIL = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_overlap_moment_source_detail.csv'
SUMMARY = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_overlap_moment_source_summary.csv'
FIG = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_overlap_moment_source_figure.png'


def main() -> None:
    detail = pd.read_csv(DETAIL)
    summary = pd.read_csv(SUMMARY).iloc[0]

    Ds = detail['D']
    theta_floor = float(summary['theta_neg_floor'])
    abs_floor = float(summary['M_band_abs_floor_abs'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    ax.plot(Ds, detail['signed_ratio'], marker='o', label=r'$r_{\rm sgn}=M_{\rm signed}/M_{\rm abs}$')
    ax.axhline(float(summary['max_signed_ratio']), color='tab:red', linestyle='--', label='ceiling')
    ax.axhline(float(summary['min_signed_ratio']), color='tab:green', linestyle='--', label='floor')
    ax.set_title('Signed-Imbalance Ratio')
    ax.set_xlabel('D')
    ax.set_ylabel('ratio')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(Ds, detail['M_band_neg'], marker='o', label=r'$M_{\rm band}^{(-)}$')
    ax.plot(Ds, detail['M_band_abs'], marker='s', label=r'$M_{\rm band}^{|\cdot|}$')
    ax.plot(Ds, theta_floor * detail['M_band_abs'], marker='^', label=r'$\theta_{\rm floor} M_{\rm band}^{|\cdot|}$')
    ax.axhline(abs_floor, color='tab:gray', linestyle=':', label=r'$\min M_{\rm band}^{|\cdot|}$')
    ax.set_title('Negative Moment vs Absolute Mass')
    ax.set_xlabel('D')
    ax.set_ylabel('moment')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(Ds, detail['theta_neg'], marker='o', label=r'$\theta_{\rm neg}=(1-r_{\rm sgn})/2$')
    ax.axhline(theta_floor, color='tab:red', linestyle='--', label=r'$\theta_{\rm floor}$')
    ax.plot(Ds, detail['neg_fraction'], marker='s', linestyle=':', label=r'$M_- / M_{|\cdot|}$')
    ax.set_title('Negative Fraction Factor')
    ax.set_xlabel('D')
    ax.set_ylabel('factor')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG, dpi=180)
    print(FIG)


if __name__ == '__main__':
    main()
