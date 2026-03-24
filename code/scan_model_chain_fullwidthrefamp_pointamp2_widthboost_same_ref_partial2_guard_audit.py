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

FOCUS_D = np.array([
    5.627118644067797,
    5.898305084745763,
    6.169491525423729,
    6.4406779661016955,
    6.711864406779661,
], dtype=float)
ETA_GRID = np.linspace(0.2, 4.0, 21)
REF_D = 9.694915254237287
REF_ETA = 1.0
LAYER = 2
OBSERVABLE_MODE = 'eft_wilson_uv_rge'
T_COH = float(PAPER_BASELINE['t_coh'])
N_MAX = int(PAPER_BASELINE['hll_observable_nmax'])
CURRENT_MODE = 'cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost'

PEAKS = (0.25, 0.5, 0.75, 1.0)
CENTERS = (5.84, 5.88, 5.90, 5.92, 5.96)
SIGMAS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.16)


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


def _local_beta(peak: float, center: float, sigma: float, D: float) -> float:
    peak = float(np.clip(peak, 0.0, 1.0))
    if peak <= 0.0:
        return 0.0
    sigma = max(float(sigma), 1e-9)
    return float(peak * np.exp(-0.5 * ((float(D) - float(center)) / sigma) ** 2))


def _geom_blend(cur: float, target: float, beta: float, floor: float) -> float:
    beta = float(np.clip(beta, 0.0, 1.0))
    if beta <= 0.0:
        return float(cur)
    return float(np.exp((1.0 - beta) * np.log(max(float(cur), floor)) + beta * np.log(max(float(target), floor))))


def _build_kinetics():
    full = make_baseline_kinetics(chain_mode='full_direct', **_common_kwargs())
    cur = make_baseline_kinetics(chain_mode=CURRENT_MODE, **_common_kwargs())
    return full, cur


def _mu(kin, D: float, eta: float) -> float:
    return float(
        kin.hll_mu_pred(
            LAYER,
            D,
            eta,
            T_COH,
            ref_D=REF_D,
            ref_eta=REF_ETA,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )


def _base_payload(kin_cur, kin_full, D: float, eta: float) -> dict[str, float]:
    target_mu = _mu(kin_full, D, eta)
    raw_amp_ref = float(kin_cur.hll_channel_amplitude(LAYER, REF_D, REF_ETA, T_COH, observable_mode=OBSERVABLE_MODE, N_max=N_MAX))
    amp_ref = float(kin_cur._blend_observable_ref_amp(raw_amp_ref, LAYER, OBSERVABLE_MODE, float(D)))
    raw_amp = float(kin_cur.hll_channel_amplitude(LAYER, float(D), float(eta), T_COH, observable_mode=OBSERVABLE_MODE, N_max=N_MAX))
    amp = float(kin_cur._blend_observable_point_amp(raw_amp, LAYER, OBSERVABLE_MODE, float(D), float(eta)))
    width = float(kin_cur.hll_total_width_ratio_uv_rge(D=float(D), eta=float(eta), t_coh=T_COH, ref_D=REF_D, ref_eta=REF_ETA, N_max=N_MAX))

    raw_amp_ref_base = float(kin_full.hll_channel_amplitude(LAYER, REF_D, REF_ETA, T_COH, observable_mode=OBSERVABLE_MODE, N_max=N_MAX))
    amp_ref_base = float(kin_full._blend_observable_ref_amp(raw_amp_ref_base, LAYER, OBSERVABLE_MODE, float(D)))
    raw_amp_base = float(kin_full.hll_channel_amplitude(LAYER, float(D), float(eta), T_COH, observable_mode=OBSERVABLE_MODE, N_max=N_MAX))
    amp_base = float(kin_full._blend_observable_point_amp(raw_amp_base, LAYER, OBSERVABLE_MODE, float(D), float(eta)))
    partial_base = float((amp_base / max(amp_ref_base, 1e-30)) ** 2)

    return {
        'target_mu': target_mu,
        'cur_partial': float((amp / max(amp_ref, 1e-30)) ** 2),
        'cur_width': width,
        'cur_mu': float((amp / max(amp_ref, 1e-30)) ** 2 / max(width, 1e-30)),
        'base_partial': partial_base,
        'floor': float(kin_cur.params.b_overlap_floor),
    }


def _build_base_detail() -> pd.DataFrame:
    kin_full, kin_cur = _build_kinetics()
    rows: list[dict[str, float]] = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            rows.append({'D': float(D), 'eta': float(eta), **_base_payload(kin_cur, kin_full, float(D), float(eta))})
    return pd.DataFrame(rows)


def _evaluate_combo(base: pd.DataFrame, peak: float, center: float, sigma: float) -> pd.DataFrame:
    df = base.copy()
    df['beta'] = [_local_beta(peak, center, sigma, d) for d in df['D']]
    df['partial_new'] = [
        _geom_blend(cp, bp, b, fl)
        for cp, bp, b, fl in zip(df['cur_partial'], df['base_partial'], df['beta'], df['floor'])
    ]
    df['mu_new'] = df['partial_new'] / df['cur_width'].clip(lower=1e-30)
    df['delta_mu'] = df['mu_new'] - df['target_mu']
    return df


def _summarize_combo(df: pd.DataFrame, peak: float, center: float, sigma: float) -> dict[str, float]:
    by_d = {float(d): float(np.max(np.abs(g['delta_mu']))) for d, g in df.groupby('D', sort=True)}
    hotspot = max(by_d.get(5.898305084745763, 0.0), by_d.get(6.169491525423729, 0.0))
    guard = max(
        0.75 * by_d.get(5.627118644067797, 0.0),
        0.5 * by_d.get(6.4406779661016955, 0.0),
        0.5 * by_d.get(6.711864406779661, 0.0),
    )
    return {
        'peak': float(peak),
        'center_D': float(center),
        'sigma_D': float(sigma),
        'p95_abs_delta_mu': float(np.percentile(np.abs(df['delta_mu']), 95.0)),
        'max_abs_delta_mu': float(np.max(np.abs(df['delta_mu']))),
        'D5p627_max_abs_delta_mu': float(by_d.get(5.627118644067797, np.nan)),
        'D5p898_max_abs_delta_mu': float(by_d.get(5.898305084745763, np.nan)),
        'D6p169_max_abs_delta_mu': float(by_d.get(6.169491525423729, np.nan)),
        'D6p441_max_abs_delta_mu': float(by_d.get(6.4406779661016955, np.nan)),
        'D6p712_max_abs_delta_mu': float(by_d.get(6.711864406779661, np.nan)),
        'hotspot_objective': float(hotspot),
        'guard_objective': float(guard),
        'joint_objective': float(max(hotspot, guard)),
    }


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    sc = axes[0].scatter(summary['D5p898_max_abs_delta_mu'], summary['D6p169_max_abs_delta_mu'], c=summary['peak'], s=38, cmap='viridis')
    axes[0].set_title('partial2 hotspot tradeoff')
    axes[0].set_xlabel('D≈5.898 max |Δμ|')
    axes[0].set_ylabel('D≈6.169 max |Δμ|')
    fig.colorbar(sc, ax=axes[0], label='peak')
    sc2 = axes[1].scatter(summary['guard_objective'], summary['joint_objective'], c=summary['center_D'], s=38, cmap='plasma')
    axes[1].set_title('guard vs joint objective')
    axes[1].set_xlabel('guard objective')
    axes[1].set_ylabel('joint objective')
    fig.colorbar(sc2, ax=axes[1], label='center_D')
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
    base = _build_base_detail()
    detail_rows = []
    summary_rows = []
    combos = [(p, c, s) for p in PEAKS for c in CENTERS for s in SIGMAS]
    for i, (peak, center, sigma) in enumerate(combos, start=1):
        print(f'[combo {i}/{len(combos)}] peak={peak:.2f} center={center:.2f} sigma={sigma:.2f}')
        df = _evaluate_combo(base, peak, center, sigma)
        keep = df[['D', 'eta', 'target_mu', 'cur_mu', 'mu_new', 'delta_mu', 'beta']].copy()
        keep.insert(0, 'sigma_D', float(sigma))
        keep.insert(0, 'center_D', float(center))
        keep.insert(0, 'peak', float(peak))
        detail_rows.append(keep)
        summary_rows.append(_summarize_combo(df, peak, center, sigma))

    detail = pd.concat(detail_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(['joint_objective', 'guard_objective', 'p95_abs_delta_mu', 'center_D', 'sigma_D']).reset_index(drop=True)
    slices = summary[['peak', 'center_D', 'sigma_D', 'D5p627_max_abs_delta_mu', 'D5p898_max_abs_delta_mu', 'D6p169_max_abs_delta_mu', 'D6p441_max_abs_delta_mu', 'D6p712_max_abs_delta_mu', 'hotspot_objective', 'guard_objective', 'joint_objective']]

    detail_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_same_ref_partial2_guard_audit_detail.csv'
    summary_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_same_ref_partial2_guard_audit_summary.csv'
    slices_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_same_ref_partial2_guard_audit_slices.csv'
    png_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_same_ref_partial2_guard_audit.png'
    meta_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_same_ref_partial2_guard_audit_run_meta.json'

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(json.dumps({
        'current_mode': CURRENT_MODE,
        'focus_D': [float(x) for x in FOCUS_D],
        'eta_grid': [float(x) for x in ETA_GRID],
        'ref_D': float(REF_D),
        'peaks': list(PEAKS),
        'centers': list(CENTERS),
        'sigmas': list(SIGMAS),
    }, indent=2))
    _mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
