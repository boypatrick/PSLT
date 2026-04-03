#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'kinetic_action_chain'
PAPERDIR = ROOT / 'paper'
OUT_TAG = os.environ.get('OUT_TAG', 'd60_d64_sobol_doe_v1')
SAMPLES = OUTDIR / f'runtime_direct_detlin_{OUT_TAG}_samples.csv'
EFFECTS = OUTDIR / f'runtime_direct_detlin_{OUT_TAG}_effects.csv'
SUMMARY = OUTDIR / f'runtime_direct_detlin_{OUT_TAG}_summary.json'
OUTPNG = OUTDIR / f'runtime_direct_detlin_{OUT_TAG}.png'


def main() -> None:
    samples = pd.read_csv(SAMPLES)
    effects = pd.read_csv(EFFECTS)
    with open(SUMMARY) as f:
        summary = json.load(f)

    current = summary['current_reference']
    best_obs = summary['best_observed']
    best_pred = summary['predicted_next_best']

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
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

    ax = axes[0, 2]
    scatter = ax.scatter(
        samples['g_center'],
        samples['g_half_width'],
        c=samples['p95_abs_delta_mu_mumu'],
        s=60,
        cmap='viridis_r',
        edgecolor='black',
        linewidth=0.3,
    )
    ax.scatter(
        [current['g_center']],
        [current['g_half_width']],
        color='#444444',
        marker='x',
        s=80,
        linewidth=2,
        label='current',
    )
    ax.scatter(
        [best_obs['g_center']],
        [best_obs['g_half_width']],
        color='#d62728',
        marker='*',
        s=150,
        label='best observed',
    )
    ax.scatter(
        [best_pred['g_center']],
        [best_pred['g_half_width']],
        color='#1f77b4',
        marker='D',
        s=60,
        label='predicted',
    )
    ax.set_xlabel('g_center')
    ax.set_ylabel('g_half_width')
    ax.set_title('Center-width valley')
    ax.legend(frameon=False, fontsize=8, loc='best')
    fig.colorbar(scatter, ax=ax, label='p95 |Δmu_mumu|')

    ax = axes[1, 0]
    beta_center = samples.pivot_table(index='g_beta', columns='g_center', values='p95_abs_delta_mu_mumu', aggfunc='min')
    im = ax.imshow(beta_center.values, aspect='auto', origin='lower', cmap='viridis_r')
    ax.set_xticks(range(len(beta_center.columns)))
    ax.set_xticklabels([f'{v:.4f}' for v in beta_center.columns], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(beta_center.index)))
    ax.set_yticklabels([f'{v:.4f}' for v in beta_center.index], fontsize=8)
    ax.set_xlabel('g_center')
    ax.set_ylabel('g_beta')
    ax.set_title('Beta-center slices')
    fig.colorbar(im, ax=ax, label='p95 |Δmu_mumu|')

    ax = axes[1, 1]
    center_width = samples.pivot_table(index='g_half_width', columns='g_center', values='p95_abs_delta_mu_mumu', aggfunc='min')
    im = ax.imshow(center_width.values, aspect='auto', origin='lower', cmap='viridis_r')
    ax.set_xticks(range(len(center_width.columns)))
    ax.set_xticklabels([f'{v:.4f}' for v in center_width.columns], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(center_width.index)))
    ax.set_yticklabels([f'{v:.4f}' for v in center_width.index], fontsize=8)
    ax.set_xlabel('g_center')
    ax.set_ylabel('g_half_width')
    ax.set_title('Center-width slices')
    fig.colorbar(im, ax=ax, label='p95 |Δmu_mumu|')

    ax = axes[1, 2]
    eff = effects[effects['term'] != 'const'].copy()
    eff['abs_coef'] = eff['coef'].abs()
    eff = eff.sort_values('abs_coef', ascending=True).tail(6)
    ax.barh(eff['term'], eff['abs_coef'], color='#9467bd')
    ax.set_xlabel('|quadratic surrogate coefficient|')
    ax.set_title('Top local effects')

    fig.suptitle(
        f'D60 D=6.4 Sobol DoE ({OUT_TAG})\n'
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
