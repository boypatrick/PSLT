#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'chi_open_system'
DETAIL = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_absolute_overlap_source_detail.csv'
SUMMARY = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_absolute_overlap_source_summary.csv'
FIG = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_absolute_overlap_source_figure.png'


def main() -> None:
    detail = pd.read_csv(DETAIL)
    summary = pd.read_csv(SUMMARY).iloc[0]
    Ds = detail['D']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    ax.plot(Ds, detail['mir_avg_w_self'], marker='o', label=r'$\bar\phi_{\rm mir|self}$')
    ax.plot(Ds, detail['self_avg_w_mir'], marker='s', label=r'$\bar\phi_{\rm self|mir}$')
    ax.set_title('Weighted Localized Averages')
    ax.set_xlabel('D')
    ax.set_ylabel('average amplitude')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(Ds, detail['M_band_abs'], marker='o', label=r'$M_{\rm band}^{\rm(abs)}$')
    ax.plot(Ds, 2.0 * detail['S_self'] * detail['mir_avg_w_self'], marker='s', linestyle='--', label=r'$2\bar\phi_{\rm mir|self} S_{\rm self}$')
    ax.plot(Ds, [float(summary['induced_self_route_abs'])] * len(Ds), linestyle=':', color='tab:red', label='self-route floor')
    ax.set_title('Absolute Overlap-Band Mass')
    ax.set_xlabel('D')
    ax.set_ylabel('moment')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(Ds, detail['S_self'], marker='o', label=r'$S_{\rm self}$')
    ax.plot(Ds, detail['S_mir'], marker='s', label=r'$S_{\rm mir}$')
    ax.set_title('Self vs Mirror Carrier Moments')
    ax.set_xlabel('D')
    ax.set_ylabel('moment')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG, dpi=180)
    print(FIG)


if __name__ == '__main__':
    main()
