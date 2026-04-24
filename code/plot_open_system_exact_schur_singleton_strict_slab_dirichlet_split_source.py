#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    s = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_dirichlet_split_source_summary.csv').iloc[0]
    rows = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_dirichlet_split_source_rows.csv')

    req_g = float(s['required_global'])
    req_l = float(s['required_local'])
    exact = float(s['strict_slab_signed_mass'])
    bdry = float(s['strict_slab_boundary_lift_mass'])
    src = float(s['strict_slab_source_potential_mass'])
    crude = float(s['crude_barrier_mass'])
    deficit = float(s['global_deficit_after_boundary'])

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    labels = ['exact slab', 'boundary lift', 'source pot.', 'crude source']
    vals = [exact, bdry, src, crude]
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    ax.bar(labels, vals, color=colors)
    ax.axhline(req_g, color='black', linestyle='--', linewidth=1.0, label='global target')
    ax.axhline(req_l, color='gray', linestyle=':', linewidth=1.5, label='local target')
    ax.set_ylabel('signed mass')
    ax.set_title(r'Dirichlet split on the strict slab')
    ax.tick_params(axis='x', rotation=18)
    ax.legend(loc='best', fontsize=9)

    ax = axes[1]
    labels = ['global deficit after boundary', 'exact source', 'crude source']
    vals = [deficit, src, crude]
    colors = ['tab:purple', 'tab:orange', 'tab:red']
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel('mass scale')
    ax.set_title(r'what actually closes the remaining gap')
    ax.tick_params(axis='x', rotation=18)

    ax = axes[2]
    ax.plot(rows['z_rel'], rows['signed_mass_exact'], marker='o', label='exact row mass')
    ax.plot(rows['z_rel'], rows['signed_mass_boundary'], marker='s', label='boundary-lift row mass')
    ax.plot(rows['z_rel'], rows['signed_mass_source'], marker='^', label='source row mass')
    ax.set_xlabel(r'$z-D/2$')
    ax.set_ylabel('row mass')
    ax.set_title(r'rowwise split inside the strict slab')
    ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_dirichlet_split_source_figure.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
