#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    s = pd.read_csv(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_edge_product_source_summary.csv'
    ).iloc[0]
    d = pd.read_csv(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_edge_product_source_detail.csv'
    )

    target = float(s['target_global_after_source'])
    physical = float(s['physical_trace_mass_product_sum'])
    required_avg = float(s['required_physical_average_poisson_weight'])
    actual_avg = float(s['physical_average_poisson_weight'])

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.bar(d['edge'], d['poisson_weighted_mass'], color=['tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:gray'])
    ax.axhline(target, color='black', linestyle='--', linewidth=1.0, label='needed physical trace')
    ax.set_ylabel('edge product mass')
    ax.set_title(r'edgewise adjoint-Poisson products')
    ax.tick_params(axis='x', rotation=18)
    ax.legend(loc='best', fontsize=9)

    ax = axes[1]
    ax.plot(d['edge'], d['cumulative_mass'], marker='o', color='tab:purple')
    ax.axhline(target, color='black', linestyle='--', linewidth=1.0, label='needed physical trace')
    ax.set_ylabel('cumulative mass')
    ax.set_title(r'outer + bottom + corners close the trace')
    ax.tick_params(axis='x', rotation=18)
    ax.legend(loc='best', fontsize=9)

    ax = axes[2]
    labels = ['actual avg weight', 'required avg weight']
    vals = [actual_avg, required_avg]
    ax.bar(labels, vals, color=['tab:blue', 'black'])
    ax.set_ylabel(r'adjoint-Poisson weight')
    ax.set_title(r'equivalent averaged-weight target')
    ax.tick_params(axis='x', rotation=18)
    ax.text(0, actual_avg, f'{physical/target:.4f}x', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_edge_product_source_figure.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
