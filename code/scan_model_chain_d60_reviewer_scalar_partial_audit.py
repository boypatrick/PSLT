#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'kinetic_action_chain'
PAPER_DIR = ROOT / 'paper'

REF_MAP = ROOT / 'output' / 'hll_signal_strength' / 'hll_signal_strength_map_chain_mode_full_direct_D60E21_tuned.csv'
CUR_MAP = ROOT / 'output' / 'hll_signal_strength' / 'hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21.csv'

HOTSPOT_CENTERS = (
    5.627118644067797,
    5.898305084745763,
    6.4406779661016955,
)
PEAKS = (0.25, 0.5, 0.75, 1.0)
SIGMAS = (0.03, 0.05, 0.08, 0.12, 0.16)


def _load_base() -> tuple[pd.DataFrame, dict[float, float]]:
    ref = pd.read_csv(REF_MAP)
    cur = pd.read_csv(CUR_MAP)
    merged = ref.merge(cur, on=['D', 'eta'], suffixes=('_ref', '_cur')).sort_values(['D', 'eta']).reset_index(drop=True)
    merged['delta_mu'] = merged['mu_mumu_cur'] - merged['mu_mumu_ref']
    scales: dict[float, float] = {}
    for center in HOTSPOT_CENTERS:
        g = merged[np.isclose(merged['D'], center)].copy()
        ratio = (g['mu_mumu_ref'] / g['mu_mumu_cur']).to_numpy(dtype=float)
        scales[float(center)] = float(np.exp(np.mean(np.log(np.clip(ratio, 1e-30, None)))))
    return merged, scales


def _local_beta(peak: float, center: float, sigma: float, D: float) -> float:
    peak = float(np.clip(peak, 0.0, 1.0))
    sigma = max(float(sigma), 1e-9)
    return float(peak * np.exp(-0.5 * ((float(D) - float(center)) / sigma) ** 2))


def _evaluate(base: pd.DataFrame, scales: dict[float, float], peak: float, sigma: float) -> pd.DataFrame:
    df = base.copy()
    log_scale = np.zeros(len(df), dtype=float)
    for center in HOTSPOT_CENTERS:
        beta = np.array([_local_beta(peak, center, sigma, d) for d in df['D']], dtype=float)
        log_scale += beta * np.log(max(scales[float(center)], 1e-30))
    df['scalar_beta_peak'] = float(peak)
    df['scalar_sigma_D'] = float(sigma)
    df['scalar_mu_factor'] = np.exp(log_scale)
    df['mu_mumu_new'] = df['mu_mumu_cur'] * df['scalar_mu_factor']
    df['delta_mu_new'] = df['mu_mumu_new'] - df['mu_mumu_ref']
    df['acceptance_mismatch_new'] = (
        ((df['chi2_mumu_ref'] <= 4.0) ^ (((df['mu_mumu_new'] - 1.4) / 0.4) ** 2 <= 4.0))
    ).astype(float)
    return df


def _summarize(df: pd.DataFrame, peak: float, sigma: float) -> dict[str, float]:
    by_d = {float(d): float(np.max(np.abs(g['delta_mu_new']))) for d, g in df.groupby('D', sort=True)}
    return {
        'peak': float(peak),
        'sigma_D': float(sigma),
        'p95_abs_delta_mu': float(np.percentile(np.abs(df['delta_mu_new']), 95.0)),
        'max_abs_delta_mu': float(np.max(np.abs(df['delta_mu_new']))),
        'acceptance_mismatch': float(np.mean(df['acceptance_mismatch_new'])),
        'D5p627_max_abs_delta_mu': float(by_d.get(5.627118644067797, np.nan)),
        'D5p898_max_abs_delta_mu': float(by_d.get(5.898305084745763, np.nan)),
        'D6p169_max_abs_delta_mu': float(by_d.get(6.169491525423729, np.nan)),
        'D6p441_max_abs_delta_mu': float(by_d.get(6.4406779661016955, np.nan)),
        'D6p712_max_abs_delta_mu': float(by_d.get(6.711864406779661, np.nan)),
        'hotspot_objective': float(max(
            by_d.get(5.627118644067797, 0.0),
            by_d.get(5.898305084745763, 0.0),
            by_d.get(6.4406779661016955, 0.0),
        )),
        'guard_objective': float(max(
            by_d.get(6.169491525423729, 0.0),
            by_d.get(6.711864406779661, 0.0),
        )),
        'joint_objective': float(max(
            max(by_d.get(5.627118644067797, 0.0), by_d.get(5.898305084745763, 0.0), by_d.get(6.4406779661016955, 0.0)),
            0.5 * max(by_d.get(6.169491525423729, 0.0), by_d.get(6.711864406779661, 0.0)),
        )),
    }


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    sc = axes[0].scatter(summary['D5p627_max_abs_delta_mu'], summary['D5p898_max_abs_delta_mu'], c=summary['peak'], s=40, cmap='viridis')
    axes[0].set_title('reviewer scalar hotspot tradeoff')
    axes[0].set_xlabel('D≈5.627 max |Δμ|')
    axes[0].set_ylabel('D≈5.898 max |Δμ|')
    fig.colorbar(sc, ax=axes[0], label='peak')
    sc2 = axes[1].scatter(summary['guard_objective'], summary['joint_objective'], c=summary['sigma_D'], s=40, cmap='plasma')
    axes[1].set_title('guard vs joint objective')
    axes[1].set_xlabel('guard objective')
    axes[1].set_ylabel('joint objective')
    fig.colorbar(sc2, ax=axes[1], label='sigma_D')
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base, scales = _load_base()
    detail_rows = []
    summary_rows = []
    combos = [(peak, sigma) for peak in PEAKS for sigma in SIGMAS]
    for i, (peak, sigma) in enumerate(combos, start=1):
        print(f'[combo {i}/{len(combos)}] peak={peak:.2f} sigma={sigma:.2f}')
        df = _evaluate(base, scales, peak, sigma)
        keep = df[['D', 'eta', 'mu_mumu_ref', 'mu_mumu_cur', 'mu_mumu_new', 'delta_mu_new', 'scalar_mu_factor', 'acceptance_mismatch_new']].copy()
        keep.insert(0, 'sigma_D', float(sigma))
        keep.insert(0, 'peak', float(peak))
        detail_rows.append(keep)
        summary_rows.append(_summarize(df, peak, sigma))

    detail = pd.concat(detail_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(['joint_objective', 'guard_objective', 'p95_abs_delta_mu', 'sigma_D']).reset_index(drop=True)
    slices = summary[['peak', 'sigma_D', 'D5p627_max_abs_delta_mu', 'D5p898_max_abs_delta_mu', 'D6p169_max_abs_delta_mu', 'D6p441_max_abs_delta_mu', 'D6p712_max_abs_delta_mu', 'hotspot_objective', 'guard_objective', 'joint_objective']]

    detail_path = OUTDIR / 'model_chain_d60_reviewer_scalar_partial_audit_detail.csv'
    summary_path = OUTDIR / 'model_chain_d60_reviewer_scalar_partial_audit_summary.csv'
    slices_path = OUTDIR / 'model_chain_d60_reviewer_scalar_partial_audit_slices.csv'
    png_path = OUTDIR / 'model_chain_d60_reviewer_scalar_partial_audit.png'
    meta_path = OUTDIR / 'model_chain_d60_reviewer_scalar_partial_audit_run_meta.json'

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(json.dumps({
        'ref_map': str(REF_MAP),
        'cur_map': str(CUR_MAP),
        'hotspot_centers': [float(x) for x in HOTSPOT_CENTERS],
        'hotspot_scales': {f'{k:.15f}': float(v) for k, v in scales.items()},
        'peaks': list(PEAKS),
        'sigmas': list(SIGMAS),
    }, indent=2))
    _mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
