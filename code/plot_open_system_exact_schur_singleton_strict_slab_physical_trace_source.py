#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    s = pd.read_csv(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_physical_trace_source_summary.csv'
    ).iloc[0]
    d = pd.read_csv(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_physical_trace_source_detail.csv'
    )

    req_g = float(s['required_global'])
    req_phys = float(s['physical_trace_required_global_after_source'])
    src = float(s['source_potential_mass'])
    phys = float(s['physical_trace_mass'])
    outer = float(s['outer_wall_mass'])
    bottom = float(s['bottom_edge_mass'])
    corners = float(s['lower_corner_mass'])

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    labels = ['outer wall', 'bottom', 'lower corners', 'axis']
    vals = [
        float(s['outer_wall_mass']),
        float(s['bottom_edge_mass']),
        float(s['lower_corner_mass']),
        float(s['axis_mass']),
    ]
    ax.bar(labels, vals, color=['tab:blue', 'tab:green', 'tab:orange', 'tab:gray'])
    ax.axhline(req_phys, color='black', linestyle='--', linewidth=1.0, label='needed after source')
    ax.set_ylabel('physical trace mass')
    ax.set_title(r'physical boundary trace components')
    ax.tick_params(axis='x', rotation=18)
    ax.legend(loc='best', fontsize=9)

    ax = axes[1]
    labels = ['outer+bottom', '+ right corner', '+ both lower corners', 'needed after source']
    vals = [
        float(s['outer_bottom_mass']),
        float(s['outer_bottom_right_corner_mass']),
        phys,
        req_phys,
    ]
    ax.bar(labels, vals, color=['tab:cyan', 'tab:purple', 'tab:blue', 'black'])
    ax.set_ylabel('physical trace mass')
    ax.set_title(r'how the physical trace crosses the needed level')
    ax.tick_params(axis='x', rotation=18)

    ax = axes[2]
    labels = ['physical trace + source', 'global target', 'source']
    vals = [phys + src, req_g, src]
    ax.bar(labels, vals, color=['tab:purple', 'black', 'tab:orange'])
    ax.set_ylabel('mass scale')
    ax.set_title(r'closing the strict-slab target')
    ax.tick_params(axis='x', rotation=18)

    fig.tight_layout()
    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_physical_trace_source_figure.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
