#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'kinetic_action_chain'
PAPER_DIR = ROOT / 'paper'

D_GRID21 = np.linspace(4.0, 20.0, 21)
D_GRID60 = np.linspace(4.0, 20.0, 60)
ETA_GRID = np.linspace(0.2, 4.0, 21)
REF_D21 = 9.6
REF_D60 = float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - 9.6)))])
REF_ETA = 1.0
LAYER = 2
OBSERVABLE_MODE = 'eft_wilson_uv_rge'
T_COH = float(PAPER_BASELINE['t_coh'])
N_MAX = int(PAPER_BASELINE['hll_observable_nmax'])

D21_FOCUS = np.array([5.6, 6.4, 7.2], dtype=float)
D60_FOCUS = np.array([
    float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - x)))])
    for x in [5.9, 6.17, 6.44]
], dtype=float)

BETA_PEAKS = (0.20, 0.35, 0.50, 0.65)
CENTER_DS = (6.35, 6.40, 6.44)
SIGMA_DS = (0.03, 0.05, 0.08, 0.10, 0.15)
GRID_CONFIGS = (
    ('D21', 21, REF_D21, D21_FOCUS),
    ('D60', 60, REF_D60, D60_FOCUS),
)


def _common_kwargs(d_num: int, d_min: float = 4.0, d_max: float = 20.0) -> dict[str, float | int | str | bool]:
    return dict(
        observable_mode=OBSERVABLE_MODE,
        d_min=float(d_min),
        d_max=float(d_max),
        d_num=int(d_num),
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
        runtime_direct_force=False,
        runtime_direct_no_cache=False,
        runtime_direct_chi_rho_max=3.0,
        runtime_direct_chi_z_margin=6.0,
        runtime_direct_chi_n_mu=120,
        runtime_direct_chi_tol=1e-8,
        runtime_direct_chi_maxiter=30000,
        runtime_direct_chi_sigma=2.5,
    )


def _mu(kin, ref_D: float, D: float, eta: float) -> tuple[float, float]:
    mu = float(
        kin.hll_mu_pred(
            LAYER,
            D=D,
            eta=eta,
            t_coh=T_COH,
            ref_D=ref_D,
            ref_eta=REF_ETA,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    chi2 = float(((mu - 1.4) / 0.4) ** 2)
    return mu, chi2


def _build_kinetics(d_num: int, beta_peak: float, center_D: float, sigma_D: float):
    common = _common_kwargs(d_num)
    return make_baseline_kinetics(
        chain_mode='cell_direct_runtime_release_fullwidthrefamp_pointamp',
        observable_point_amp_anchor_peak_override=float(beta_peak),
        observable_point_amp_anchor_center_D_override=float(center_D),
        observable_point_amp_anchor_sigma_D_override=float(sigma_D),
        **common,
    )


def _build_full_kinetics(d_num: int):
    return make_baseline_kinetics(chain_mode='full_direct', **_common_kwargs(d_num))


def _precompute_baseline_grid(grid_name: str, d_num: int, ref_D: float, focus_D: np.ndarray) -> list[dict[str, float | str]]:
    kin_full = _build_full_kinetics(d_num)
    rows: list[dict[str, float | str]] = []
    for D in focus_D:
        for eta in ETA_GRID:
            base_mu, base_chi2 = _mu(kin_full, ref_D, float(D), float(eta))
            rows.append({
                'grid': grid_name,
                'D': float(D),
                'eta': float(eta),
                'base_mu': float(base_mu),
                'base_chi2': float(base_chi2),
            })
    return rows


def _evaluate_grid(
    baseline_rows: list[dict[str, float | str]],
    d_num: int,
    ref_D: float,
    beta_peak: float,
    center_D: float,
    sigma_D: float,
) -> list[dict[str, float | str]]:
    kin_cur = _build_kinetics(d_num, beta_peak, center_D, sigma_D)
    rows: list[dict[str, float | str]] = []
    for base in baseline_rows:
        D = float(base['D'])
        eta = float(base['eta'])
        base_mu = float(base['base_mu'])
        base_chi2 = float(base['base_chi2'])
        cur_mu, cur_chi2 = _mu(kin_cur, ref_D, D, eta)
        rows.append({
            'grid': str(base['grid']),
            'beta_peak': float(beta_peak),
            'center_D': float(center_D),
            'sigma_D': float(sigma_D),
            'D': D,
            'eta': eta,
            'delta_mu': float(cur_mu - base_mu),
            'abs_delta_mu': float(abs(cur_mu - base_mu)),
            'acceptance_mismatch': float((base_chi2 <= 4.0) ^ (cur_chi2 <= 4.0)),
        })
    return rows


def _build_detail() -> pd.DataFrame:
    baseline_cache = {
        grid_name: _precompute_baseline_grid(grid_name, d_num, ref_D, focus_D)
        for grid_name, d_num, ref_D, focus_D in GRID_CONFIGS
    }
    rows: list[dict[str, float | str]] = []
    combos = [(beta_peak, center_D, sigma_D) for beta_peak in BETA_PEAKS for center_D in CENTER_DS for sigma_D in SIGMA_DS]
    for idx, (beta_peak, center_D, sigma_D) in enumerate(combos, start=1):
        print(
            f'[combo {idx}/{len(combos)}] '
            f'beta={beta_peak:.2f} center_D={center_D:.2f} sigma_D={sigma_D:.2f}',
            flush=True,
        )
        for grid_name, d_num, ref_D, _focus_D in GRID_CONFIGS:
            rows.extend(
                _evaluate_grid(
                    baseline_cache[grid_name],
                    d_num,
                    ref_D,
                    beta_peak,
                    center_D,
                    sigma_D,
                )
            )
    return pd.DataFrame(rows)


def _summary_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (beta_peak, center_D, sigma_D), sub in detail.groupby(['beta_peak', 'center_D', 'sigma_D'], sort=True):
        row = {
            'beta_peak': float(beta_peak),
            'center_D': float(center_D),
            'sigma_D': float(sigma_D),
        }
        for grid_name, gsub in sub.groupby('grid', sort=True):
            row[f'{grid_name}_p95_abs_delta_mu'] = float(np.percentile(gsub['abs_delta_mu'], 95.0))
            row[f'{grid_name}_max_abs_delta_mu'] = float(np.max(gsub['abs_delta_mu']))
            row[f'{grid_name}_acceptance_mismatch'] = float(np.mean(gsub['acceptance_mismatch']))
            for d_val, dsub in gsub.groupby('D', sort=True):
                key = f'{grid_name}_D{float(d_val):.3f}'.replace('.', 'p')
                row[f'{key}_max_abs_delta_mu'] = float(np.max(dsub['abs_delta_mu']))
        row['joint_objective'] = float(
            row.get('D21_max_abs_delta_mu', np.inf)
            + row.get('D60_max_abs_delta_mu', np.inf)
            + 0.5 * row.get('D21_p95_abs_delta_mu', np.inf)
            + 0.5 * row.get('D60_p95_abs_delta_mu', np.inf)
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values('joint_objective').reset_index(drop=True)


def _slice_rows(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (beta_peak, center_D, sigma_D, grid_name, d_val), grp in detail.groupby(['beta_peak', 'center_D', 'sigma_D', 'grid', 'D'], sort=True):
        rows.append({
            'beta_peak': float(beta_peak),
            'center_D': float(center_D),
            'sigma_D': float(sigma_D),
            'grid': str(grid_name),
            'D': float(d_val),
            'slice_p95_abs_delta_mu': float(np.percentile(grp['abs_delta_mu'], 95.0)),
            'slice_max_abs_delta_mu': float(np.max(grp['abs_delta_mu'])),
            'slice_acceptance_mismatch': float(np.mean(grp['acceptance_mismatch'])),
        })
    return pd.DataFrame(rows).sort_values(['beta_peak', 'center_D', 'sigma_D', 'grid', 'D']).reset_index(drop=True)


def _plot(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (metric, title) in zip(
        axes,
        [
            ('D21_max_abs_delta_mu', 'D21 max |Δμ|'),
            ('D60_max_abs_delta_mu', 'D60 max |Δμ|'),
            ('joint_objective', 'joint objective'),
        ],
    ):
        pivot = (
            summary[summary['sigma_D'] == 0.10]
            .pivot_table(index='beta_peak', columns='center_D', values=metric, aggfunc='min')
            .sort_index()
        )
        im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'{title}\n(sigma=0.10)')
        ax.set_xlabel('center_D')
        ax.set_ylabel('beta_peak')
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f'{x:.2f}' for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f'{x:.2f}' for x in pivot.index])
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail = _build_detail()
    summary = _summary_rows(detail)
    slices = _slice_rows(detail)

    detail_path = OUTDIR / 'model_chain_pointamp_gridaware_audit_detail.csv'
    summary_path = OUTDIR / 'model_chain_pointamp_gridaware_audit_summary.csv'
    slices_path = OUTDIR / 'model_chain_pointamp_gridaware_audit_slices.csv'
    png_path = OUTDIR / 'model_chain_pointamp_gridaware_audit.png'
    meta_path = OUTDIR / 'model_chain_pointamp_gridaware_audit_run_meta.json'

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(json.dumps({
        'D21_focus': [float(x) for x in D21_FOCUS],
        'D60_focus': [float(x) for x in D60_FOCUS],
        'eta_grid': [float(x) for x in ETA_GRID],
        'beta_peaks': [float(x) for x in BETA_PEAKS],
        'center_Ds': [float(x) for x in CENTER_DS],
        'sigma_Ds': [float(x) for x in SIGMA_DS],
        'ref_D21': float(REF_D21),
        'ref_D60': float(REF_D60),
    }, indent=2))
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
