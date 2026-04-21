#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'chi_open_system'
DETAIL = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_mirror_average_source_detail.csv'
SUMMARY = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_mirror_average_source_summary.csv'
FIG = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_mirror_average_source_figure.png'


def main() -> None:
    detail = pd.read_csv(DETAIL)
    summary = pd.read_csv(SUMMARY).iloc[0]
    Ds = detail['D']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    ax.plot(Ds, detail['self_bias'], marker='o', label=r'$\Xi_{\rm band}^{(self)}$')
    ax.axhline(float(summary['self_bias_min']), color='tab:red', linestyle='--', label='bias floor')
    ax.axhline(float(summary['self_bias_max']), color='tab:green', linestyle='--', label='bias ceiling')
    ax.set_title('Self-Weight Bias Factor')
    ax.set_xlabel('D')
    ax.set_ylabel('bias')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(Ds, detail['mirror_avg_weighted'], marker='o', label=r'$\bar\Phi_{\rm band}^{(mir|self)}$')
    ax.plot(Ds, detail['mirror_avg'], marker='s', label=r'$\bar\Phi_{\rm band}^{(mir)}$')
    ax.plot(Ds, detail['self_bias'] * detail['mirror_avg'], marker='^', linestyle='--', label='bias × mean')
    ax.set_title('Weighted vs Ordinary Mirror Mean')
    ax.set_xlabel('D')
    ax.set_ylabel('amplitude')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(Ds, detail['A_band'], marker='o', label=r'$A_{\rm band}$')
    ax.plot(Ds, detail['S_mir'], marker='s', label=r'$S_{\rm band}^{(mir)}$')
    ax.set_title('Band Area and Mirror Carrier')
    ax.set_xlabel('D')
    ax.set_ylabel('measure / moment')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG, dpi=180)
    print(FIG)


if __name__ == '__main__':
    main()
