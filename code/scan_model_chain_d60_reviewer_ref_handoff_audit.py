#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, load_observations, make_baseline_kinetics

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
OBS = load_observations().get('mumu')

REVIEWER_REF_D = 9.966101694915254
SAME_REF_D = 9.694915254237287
BETA_PEAKS = (0.25, 0.5, 0.75, 1.0)
CENTER_DS = (5.56, 5.627118644067797, 5.70)
SIGMA_DS = (0.03, 0.05, 0.08)


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
    dvals = np.asarray(df['D'], dtype=float)
    snapped = []
    for d in dvals:
        idx = int(np.argmin(np.abs(FOCUS_D - float(d))))
        if abs(float(FOCUS_D[idx]) - float(d)) <= 1e-9:
            snapped.append(float(FOCUS_D[idx]))
        else:
            snapped.append(float(d))
    df['D'] = np.asarray(snapped, dtype=float)
    return df[['D', 'eta', 'mu_mumu', 'chi2_mumu']].rename(columns={'mu_mumu': 'target_mu', 'chi2_mumu': 'target_chi2'})


def _build_kinetics():
    return make_baseline_kinetics(chain_mode=BASE_MODE, **_common_kwargs())


def _handoff_ref_D(D: float, beta_peak: float, center_D: float, sigma_D: float) -> float:
    beta = float(beta_peak) * np.exp(-0.5 * ((float(D) - float(center_D)) / max(float(sigma_D), 1e-9)) ** 2)
    beta = float(np.clip(beta, 0.0, 1.0))
    return float(REVIEWER_REF_D + beta * (SAME_REF_D - REVIEWER_REF_D))


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
    if OBS is None:
        raise RuntimeError('mumu observation is unavailable')
    chi2 = float(((mu - float(OBS.mu_obs)) / float(OBS.sigma_obs)) ** 2)
    return mu, chi2


def _evaluate_combo(target: pd.DataFrame, beta_peak: float, center_D: float, sigma_D: float) -> pd.DataFrame:
    kin = _build_kinetics()
    rows = []
    for D in FOCUS_D:
        ref_D_eff = _handoff_ref_D(float(D), beta_peak, center_D, sigma_D)
        for eta in ETA_GRID:
            mu, chi2 = _mu(kin, float(D), float(eta), ref_D_eff)
            rows.append(
                {
                    'beta_peak': float(beta_peak),
                    'center_D': float(center_D),
                    'sigma_D': float(sigma_D),
                    'D': float(D),
                    'eta': float(eta),
                    'ref_D_eff': float(ref_D_eff),
                    'mu_mumu': mu,
                    'chi2_mumu': chi2,
                }
            )
    df = pd.DataFrame(rows).merge(target, on=['D', 'eta'], how='left')
    df['abs_delta_mu'] = np.abs(df['mu_mumu'] - df['target_mu'])
    df['abs_delta_chi2'] = np.abs(df['chi2_mumu'] - df['target_chi2'])
    return df


def _summarize_combo(df: pd.DataFrame, beta_peak: float, center_D: float, sigma_D: float) -> dict[str, float]:
    by_D = df.groupby('D')['abs_delta_mu'].max().to_dict()
    mismatch = float(((df['chi2_mumu'] < 4.0) ^ (df['target_chi2'] < 4.0)).mean())
    return {
        'beta_peak': float(beta_peak),
        'center_D': float(center_D),
        'sigma_D': float(sigma_D),
        'p95_abs_delta_mu': float(df['abs_delta_mu'].quantile(0.95)),
        'max_abs_delta_mu': float(df['abs_delta_mu'].max()),
        'acceptance_mismatch': mismatch,
        'D5p627_max_abs_delta_mu': float(by_D.get(FOCUS_D[0], np.nan)),
        'D5p898_max_abs_delta_mu': float(by_D.get(FOCUS_D[1], np.nan)),
        'D6p169_max_abs_delta_mu': float(by_D.get(FOCUS_D[2], np.nan)),
        'D6p441_max_abs_delta_mu': float(by_D.get(FOCUS_D[3], np.nan)),
        'D6p712_max_abs_delta_mu': float(by_D.get(FOCUS_D[4], np.nan)),
        'hotspot_objective': float(max(by_D.get(FOCUS_D[0], 0.0), by_D.get(FOCUS_D[1], 0.0), by_D.get(FOCUS_D[2], 0.0))),
        'guard_objective': float(max(by_D.get(FOCUS_D[3], 0.0), by_D.get(FOCUS_D[4], 0.0))),
        'joint_objective': float(max(max(by_D.get(FOCUS_D[0], 0.0), by_D.get(FOCUS_D[1], 0.0), by_D.get(FOCUS_D[2], 0.0)), max(by_D.get(FOCUS_D[3], 0.0), by_D.get(FOCUS_D[4], 0.0)))),
    }


def _plot(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    sc = axes[0].scatter(summary['D5p627_max_abs_delta_mu'], summary['D5p898_max_abs_delta_mu'], c=summary['beta_peak'], cmap='viridis', s=40)
    axes[0].set_xlabel('D≈5.627 max |Δμ|')
    axes[0].set_ylabel('D≈5.898 max |Δμ|')
    axes[0].set_title('Reviewer Handoff vs Intrinsic')
    fig.colorbar(sc, ax=axes[0], label='beta_peak')

    sc2 = axes[1].scatter(summary['D6p169_max_abs_delta_mu'], summary['joint_objective'], c=summary['center_D'], cmap='plasma', s=40)
    axes[1].set_xlabel('D≈6.169 max |Δμ|')
    axes[1].set_ylabel('joint objective')
    axes[1].set_title('Center Tradeoff')
    fig.colorbar(sc2, ax=axes[1], label='center_D')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    target = _load_target_map()
    combos = [(b, c, s) for b in BETA_PEAKS for c in CENTER_DS for s in SIGMA_DS]
    detail_frames = []
    summary_rows = []
    for i, (beta, center_D, sigma_D) in enumerate(combos, start=1):
        print(f'[combo {i}/{len(combos)}] beta={beta:.2f} center_D={center_D:.12f} sigma_D={sigma_D:.2f}')
        df = _evaluate_combo(target, beta, center_D, sigma_D)
        keep = df[['D','eta','ref_D_eff','mu_mumu','target_mu','chi2_mumu','target_chi2','abs_delta_mu','abs_delta_chi2']].copy()
        keep.insert(0, 'sigma_D', float(sigma_D))
        keep.insert(0, 'center_D', float(center_D))
        keep.insert(0, 'beta_peak', float(beta))
        detail_frames.append(keep)
        summary_rows.append(_summarize_combo(df, beta, center_D, sigma_D))

    detail = pd.concat(detail_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(['joint_objective','guard_objective','p95_abs_delta_mu','beta_peak']).reset_index(drop=True)
    slices = summary[['beta_peak','center_D','sigma_D','D5p627_max_abs_delta_mu','D5p898_max_abs_delta_mu','D6p169_max_abs_delta_mu','D6p441_max_abs_delta_mu','D6p712_max_abs_delta_mu','hotspot_objective','guard_objective','joint_objective']]

    detail_path = OUTDIR / 'model_chain_d60_reviewer_ref_handoff_audit_detail.csv'
    summary_path = OUTDIR / 'model_chain_d60_reviewer_ref_handoff_audit_summary.csv'
    slices_path = OUTDIR / 'model_chain_d60_reviewer_ref_handoff_audit_slices.csv'
    png_path = OUTDIR / 'model_chain_d60_reviewer_ref_handoff_audit.png'
    meta_path = OUTDIR / 'model_chain_d60_reviewer_ref_handoff_audit_run_meta.json'

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(json.dumps({
        'target_map': str(TARGET_MAP),
        'base_mode': BASE_MODE,
        'reviewer_ref_D': REVIEWER_REF_D,
        'same_ref_D': SAME_REF_D,
        'focus_D': [float(x) for x in FOCUS_D],
        'eta_grid': [float(x) for x in ETA_GRID],
        'beta_peaks': list(BETA_PEAKS),
        'center_Ds': list(CENTER_DS),
        'sigma_Ds': list(SIGMA_DS),
    }, indent=2))
    _mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
