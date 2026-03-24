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
TARGET_MAP = ROOT / 'output' / 'hll_signal_strength' / 'hll_signal_strength_map_chain_mode_full_direct_D60E21_tuned.csv'

FOCUS_D = np.array([
    5.627118644067797,
    5.898305084745763,
    6.169491525423729,
    6.4406779661016955,
    6.711864406779661,
], dtype=float)
ETA_GRID = np.linspace(0.2, 4.0, 21)
REF_ETA = 1.0
LAYER = 2
OBSERVABLE_MODE = 'eft_wilson_uv_rge'
T_COH = float(PAPER_BASELINE['t_coh'])
N_MAX = int(PAPER_BASELINE['hll_observable_nmax'])
BASE_MODE = 'cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost'

REF_DS = (
    9.694915254237287,
    9.83050847457627,
    9.966101694915254,
)
REFAMP2_PEAKS = (0.0, 0.5, 1.0)
CENTER_DS = (5.90, 6.03, 6.17)
SIGMA_DS = (0.10, 0.15, 0.20)


def _common_kwargs() -> dict[str, float | int | str | bool]:
    return dict(
        observable_mode=OBSERVABLE_MODE,
        d_min=4.0,
        d_max=20.0,
        d_num=60,
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


def _load_target_map() -> pd.DataFrame:
    df = pd.read_csv(TARGET_MAP)
    mask = np.isclose(np.asarray(df['D'], dtype=float)[:, None], FOCUS_D[None, :], rtol=0.0, atol=1e-9).any(axis=1)
    df = df[mask].copy()
    return df[['D', 'eta', 'mu_mumu', 'chi2_mumu']].rename(columns={'mu_mumu': 'target_mu', 'chi2_mumu': 'target_chi2'})


def _build_kinetics(refamp2_peak: float, center_D: float, sigma_D: float):
    return make_baseline_kinetics(
        chain_mode=BASE_MODE,
        observable_ref_amp_anchor_peak2_override=float(refamp2_peak),
        observable_ref_amp_anchor_center_D2_override=float(center_D),
        observable_ref_amp_anchor_sigma_D2_override=float(sigma_D),
        **_common_kwargs(),
    )


def _mu(kin, D: float, eta: float, ref_D: float) -> tuple[float, float]:
    mu = float(
        kin.hll_mu_pred(
            LAYER,
            D=float(D),
            eta=float(eta),
            t_coh=T_COH,
            ref_D=float(ref_D),
            ref_eta=REF_ETA,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )
    chi2 = float(((mu - 1.4) / 0.4) ** 2)
    return mu, chi2


def _evaluate_combo(target: pd.DataFrame, ref_D: float, refamp2_peak: float, center_D: float, sigma_D: float) -> pd.DataFrame:
    kin = _build_kinetics(refamp2_peak, center_D, sigma_D)
    rows: list[dict[str, float]] = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            row_mask = np.isclose(target['D'], float(D), rtol=0.0, atol=1e-9) & np.isclose(target['eta'], float(eta), rtol=0.0, atol=1e-9)
            row = target.loc[row_mask].iloc[0]
            mu, chi2 = _mu(kin, float(D), float(eta), float(ref_D))
            rows.append({
                'ref_D': float(ref_D),
                'refamp2_peak': float(refamp2_peak),
                'center_D': float(center_D),
                'sigma_D': float(sigma_D),
                'D': float(D),
                'eta': float(eta),
                'target_mu': float(row['target_mu']),
                'target_chi2': float(row['target_chi2']),
                'mu_new': float(mu),
                'delta_mu': float(mu - float(row['target_mu'])),
                'acceptance_mismatch': float((float(row['target_chi2']) <= 4.0) ^ (chi2 <= 4.0)),
            })
    return pd.DataFrame(rows)


def _summarize_combo(df: pd.DataFrame, ref_D: float, refamp2_peak: float, center_D: float, sigma_D: float) -> dict[str, float]:
    def lookup(dval: float) -> float:
        g = df[np.isclose(df['D'], dval, rtol=0.0, atol=1e-9)]
        return float(np.max(np.abs(g['delta_mu'])))

    d5627 = lookup(5.627118644067797)
    d5898 = lookup(5.898305084745763)
    d6169 = lookup(6.169491525423729)
    d6441 = lookup(6.4406779661016955)
    d6712 = lookup(6.711864406779661)
    hotspot = max(d5627, d5898, d6169)
    guard = max(d6441, d6712)
    return {
        'ref_D': float(ref_D),
        'refamp2_peak': float(refamp2_peak),
        'center_D': float(center_D),
        'sigma_D': float(sigma_D),
        'p95_abs_delta_mu': float(np.percentile(np.abs(df['delta_mu']), 95.0)),
        'max_abs_delta_mu': float(np.max(np.abs(df['delta_mu']))),
        'acceptance_mismatch': float(np.mean(df['acceptance_mismatch'])),
        'D5p627_max_abs_delta_mu': d5627,
        'D5p898_max_abs_delta_mu': d5898,
        'D6p169_max_abs_delta_mu': d6169,
        'D6p441_max_abs_delta_mu': d6441,
        'D6p712_max_abs_delta_mu': d6712,
        'hotspot_objective': float(hotspot),
        'guard_objective': float(guard),
        'joint_objective': float(max(hotspot, 0.5 * guard)),
    }


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    sc = axes[0].scatter(summary['D5p627_max_abs_delta_mu'], summary['D5p898_max_abs_delta_mu'], c=summary['ref_D'], cmap='viridis', s=36)
    axes[0].set_xlabel('D≈5.627 max |Δμ|')
    axes[0].set_ylabel('D≈5.898 max |Δμ|')
    axes[0].set_title('anchor-sensitive vs intrinsic')
    fig.colorbar(sc, ax=axes[0], label='ref_D')

    sc2 = axes[1].scatter(summary['D6p169_max_abs_delta_mu'], summary['joint_objective'], c=summary['refamp2_peak'], cmap='plasma', s=36)
    axes[1].set_xlabel('D≈6.169 max |Δμ|')
    axes[1].set_ylabel('joint objective')
    axes[1].set_title('refamp2 leverage')
    fig.colorbar(sc2, ax=axes[1], label='refamp2 peak')
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _write_mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    target = _load_target_map()
    detail_rows = []
    summary_rows = []
    combos = [(r, p, c, s) for r in REF_DS for p in REFAMP2_PEAKS for c in CENTER_DS for s in SIGMA_DS]
    for i, (ref_D, peak, center_D, sigma_D) in enumerate(combos, start=1):
        print(f'[combo {i}/{len(combos)}] ref_D={ref_D:.12f} peak={peak:.2f} center_D={center_D:.2f} sigma_D={sigma_D:.2f}')
        df = _evaluate_combo(target, ref_D, peak, center_D, sigma_D)
        keep = df[['D', 'eta', 'target_mu', 'mu_new', 'delta_mu', 'acceptance_mismatch']].copy()
        keep.insert(0, 'sigma_D', float(sigma_D))
        keep.insert(0, 'center_D', float(center_D))
        keep.insert(0, 'refamp2_peak', float(peak))
        keep.insert(0, 'ref_D', float(ref_D))
        detail_rows.append(keep)
        summary_rows.append(_summarize_combo(df, ref_D, peak, center_D, sigma_D))
    detail = pd.concat(detail_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(['joint_objective', 'guard_objective', 'p95_abs_delta_mu', 'ref_D']).reset_index(drop=True)
    slices = summary[['ref_D', 'refamp2_peak', 'center_D', 'sigma_D', 'D5p627_max_abs_delta_mu', 'D5p898_max_abs_delta_mu', 'D6p169_max_abs_delta_mu', 'D6p441_max_abs_delta_mu', 'D6p712_max_abs_delta_mu', 'hotspot_objective', 'guard_objective', 'joint_objective']]

    detail_path = OUTDIR / 'model_chain_d60_reviewer_anchor_aware_refamp_audit_detail.csv'
    summary_path = OUTDIR / 'model_chain_d60_reviewer_anchor_aware_refamp_audit_summary.csv'
    slices_path = OUTDIR / 'model_chain_d60_reviewer_anchor_aware_refamp_audit_slices.csv'
    png_path = OUTDIR / 'model_chain_d60_reviewer_anchor_aware_refamp_audit.png'
    meta_path = OUTDIR / 'model_chain_d60_reviewer_anchor_aware_refamp_audit_run_meta.json'

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(json.dumps({'target_map': str(TARGET_MAP), 'focus_D': [float(x) for x in FOCUS_D], 'eta_grid': [float(x) for x in ETA_GRID], 'ref_Ds': list(REF_DS), 'refamp2_peaks': list(REFAMP2_PEAKS), 'center_Ds': list(CENTER_DS), 'sigma_Ds': list(SIGMA_DS)}, indent=2))
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
