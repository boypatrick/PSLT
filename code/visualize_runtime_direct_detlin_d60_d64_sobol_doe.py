#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'kinetic_action_chain'
PAPERDIR = ROOT / 'paper'
SAMPLES = OUTDIR / 'runtime_direct_detlin_d60_d64_sobol_doe_v1_samples.csv'
EFFECTS = OUTDIR / 'runtime_direct_detlin_d60_d64_sobol_doe_v1_effects.csv'
SUMMARY = OUTDIR / 'runtime_direct_detlin_d60_d64_sobol_doe_v1_summary.json'
OUTPNG = OUTDIR / 'runtime_direct_detlin_d60_d64_sobol_doe_v1.png'


def main() -> None:
    samples = pd.read_csv(SAMPLES)
    effects = pd.read_csv(EFFECTS)
    with open(SUMMARY) as f:
        summary = json.load(f)

    current = summary['current_reference']
    best_obs = summary['best_observed']
    best_pred = summary['predicted_next_best']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.scatter(samples['g_beta'], samples['p95_abs_delta_mu_mumu'], s=35, alpha=0.8, color='#1f77b4')
    ax.axvline(current['g_beta'], color='#444444', linestyle='--', linewidth=1, label='current')
    ax.axvline(best_pred['g_beta'], color='#d62728', linestyle=':', linewidth=1.5, label='predicted')
    ax.set_xlabel('g_beta')
    ax.set_ylabel('p95 |Δmu_mumu|')
    ax.set_title('Beta vs residual')
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.scatter(samples['g_center'], samples['p95_abs_delta_mu_mumu'], s=35, alpha=0.8, color='#2ca02c')
    ax.axvline(current['g_center'], color='#444444', linestyle='--', linewidth=1)
    ax.axvline(best_pred['g_center'], color='#d62728', linestyle=':', linewidth=1.5)
    ax.set_xlabel('g_center')
    ax.set_ylabel('p95 |Δmu_mumu|')
    ax.set_title('Center vs residual')

    ax = axes[1, 0]
    ax.scatter(samples['g_half_width'], samples['p95_abs_delta_mu_mumu'], s=35, alpha=0.8, color='#ff7f0e')
    ax.axvline(current['g_half_width'], color='#444444', linestyle='--', linewidth=1)
    ax.axvline(best_pred['g_half_width'], color='#d62728', linestyle=':', linewidth=1.5)
    ax.set_xlabel('g_half_width')
    ax.set_ylabel('p95 |Δmu_mumu|')
    ax.set_title('Half-width vs residual')

    ax = axes[1, 1]
    eff = effects[effects['term'] != 'const'].copy()
    eff['abs_coef'] = eff['coef'].abs()
    eff = eff.sort_values('abs_coef', ascending=True).tail(6)
    ax.barh(eff['term'], eff['abs_coef'], color='#9467bd')
    ax.set_xlabel('|quadratic surrogate coefficient|')
    ax.set_title('Top local effects')

    fig.suptitle(
        'D60 D=6.4 Sobol DoE around v5\n'
        f"current=({current['g_beta']:.3f}, {current['g_center']:.3f}, {current['g_half_width']:.3f}), "
        f"best observed p95={best_obs['p95_abs_delta_mu_mumu']:.6f}, predicted next p95={best_pred['pred_p95']:.6f}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUTPNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPNG, dpi=180, bbox_inches='tight')
    plt.close(fig)
    (PAPERDIR / OUTPNG.name).write_bytes(OUTPNG.read_bytes())
    print(OUTPNG)


if __name__ == '__main__':
    main()
