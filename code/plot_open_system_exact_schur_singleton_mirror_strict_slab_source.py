#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    summary = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_strict_slab_source_summary.csv').iloc[0]
    scan = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_strict_slab_source_scan.csv').sort_values('h_cap')

    h_star = float(summary['strict_cap'])
    req_global = float(summary['required_global'])
    req_local = float(summary['required_local'])
    I_lower = float(summary['lower_half_signed_mass'])
    I_strict = float(summary['strict_slab_signed_mass'])
    I_strip = float(summary['omitted_interface_strip_mass'])
    I_bar = float(summary['crude_torsion_barrier_mass'])

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.plot(scan['h_cap'], scan['signed_mass_over_required_global'], marker='o', label=r'$I_\sharp / I_{\rm req,g}$')
    ax.plot(scan['h_cap'], scan['signed_mass_over_required_local'], marker='s', label=r'$I_\sharp / I_{\rm req,l}$')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax.axvline(h_star, color='tab:red', linestyle=':', linewidth=1.5, label=fr'$h_*= {h_star:.2f}$')
    ax.set_xlabel(r'strict cap $h$ in $z \leq D/2-h$')
    ax.set_ylabel('threshold ratio')
    ax.set_title(r'signed strict-slab mass vs. cut depth')
    ax.legend(loc='best', fontsize=9)

    ax = axes[1]
    ax.plot(scan['h_cap'], scan['min_rhs0'], marker='o', label=r'$f_{\sharp,\min}$')
    ax.plot(scan['h_cap'], scan['kappa_needed'], marker='s', label=r'needed shift $\kappa_h$')
    ax.axhline(0.0, color='black', linestyle='--', linewidth=1.0)
    ax.axvline(h_star, color='tab:red', linestyle=':', linewidth=1.5)
    ax.set_xlabel(r'strict cap $h$')
    ax.set_ylabel('value')
    ax.set_title(r'where bare $-\Delta_a$ becomes barrier-compatible')
    ax.legend(loc='best', fontsize=9)

    ax = axes[2]
    labels = ['lower half', 'strict slab', 'strip', 'crude barrier']
    vals = [I_lower, I_strict, I_strip, I_bar]
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    ax.bar(labels, vals, color=colors)
    ax.axhline(req_global, color='black', linestyle='--', linewidth=1.0, label='global target')
    ax.axhline(req_local, color='gray', linestyle=':', linewidth=1.5, label='local target')
    ax.set_ylabel('signed mass')
    ax.set_title(r'final geometry: strict slab carries the target')
    ax.legend(loc='best', fontsize=9)
    ax.tick_params(axis='x', rotation=18)

    fig.tight_layout()
    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_strict_slab_source_figure.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
