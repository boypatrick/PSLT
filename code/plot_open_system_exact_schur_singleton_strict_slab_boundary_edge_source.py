#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    s = pd.read_csv(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_boundary_edge_source_summary.csv'
    ).iloc[0]
    d = pd.read_csv(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_boundary_edge_source_detail.csv'
    )

    req_g = float(s['required_global'])
    req_l = float(s['required_local'])
    i_boundary = float(s['strict_slab_boundary_lift_mass'])
    i_source = float(s['strict_slab_source_potential_mass'])
    i_physical = float(s['boundary_mass_physical_edges'])
    i_top = float(s['boundary_mass_top_cap'])
    i_outer_bottom = float(s['boundary_mass_outer_bottom_axis'])

    categories = d[d['category'].isin(['axis', 'outer_rho', 'bottom', 'top', 'corner_lb', 'corner_lt', 'corner_rb', 'corner_rt'])]

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.bar(categories['category'], categories['harmonic_lift_mass'], color='tab:blue')
    ax.set_ylabel('harmonic-lift mass')
    ax.set_title('strict-slab boundary lift by edge type')
    ax.tick_params(axis='x', rotation=30)

    ax = axes[1]
    labels = ['boundary total', 'physical edges', 'top cap', 'source']
    vals = [i_boundary, i_physical, i_top, i_source]
    colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:orange']
    ax.bar(labels, vals, color=colors)
    ax.axhline(req_g, color='black', linestyle='--', linewidth=1.0, label='global target')
    ax.axhline(req_l, color='gray', linestyle=':', linewidth=1.5, label='local target')
    ax.set_ylabel('mass scale')
    ax.set_title('boundary split versus thresholds')
    ax.tick_params(axis='x', rotation=18)
    ax.legend(loc='best', fontsize=9)

    ax = axes[2]
    labels = ['physical+source', 'outer+bottom+source', 'global target']
    vals = [i_physical + i_source, i_outer_bottom + i_source, req_g]
    colors = ['tab:purple', 'tab:cyan', 'black']
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel('mass scale')
    ax.set_title('what still matters after removing top cap')
    ax.tick_params(axis='x', rotation=18)

    fig.tight_layout()
    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_boundary_edge_source_figure.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
