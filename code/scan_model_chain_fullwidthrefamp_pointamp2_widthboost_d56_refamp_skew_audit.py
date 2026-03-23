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
BASE_MAP = ROOT / 'output' / 'hll_signal_strength' / 'hll_signal_strength_map_chain_mode_full_direct_D60E21_tuned.csv'

FOCUS_D = np.array([
    5.627118644067797,
    5.898305084745763,
    6.169491525423729,
    6.4406779661016955,
    6.711864406779661,
], dtype=float)
ETA_GRID = np.linspace(0.2, 4.0, 21)
REF_D = 9.694915254237289
REF_ETA = 1.0
LAYER = 2
OBSERVABLE_MODE = 'eft_wilson_uv_rge'
T_COH = float(PAPER_BASELINE['t_coh'])
N_MAX = int(PAPER_BASELINE['hll_observable_nmax'])

TAPER_PEAKS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
CENTER_DS = (5.70, 5.76, 5.82)
SIGMA_DS = (0.08, 0.12, 0.16)


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


def _local_skew(D: float, center: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-9)
    x = (float(D) - float(center)) / sigma
    return float(-x * np.exp(0.5 - 0.5 * x * x))


def _geom_blend(cur: float, target: float, beta: float, floor: float) -> float:
    beta = float(np.clip(beta, 0.0, 1.0))
    return float(np.exp((1.0 - beta) * np.log(max(float(cur), floor)) + beta * np.log(max(float(target), floor))))


def _build_current_kinetics():
    return make_baseline_kinetics(
        chain_mode='cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost',
        **_common_kwargs(),
    )


def _load_target_map() -> pd.DataFrame:
    df = pd.read_csv(BASE_MAP)
    mask = np.isclose(np.asarray(df['D'], dtype=float)[:, None], FOCUS_D[None, :], rtol=0.0, atol=1e-9).any(axis=1)
    df = df[mask].copy()
    return df[['D', 'eta', 'mu_mumu', 'chi2_mumu']].rename(columns={'mu_mumu': 'target_mu', 'chi2_mumu': 'target_chi2'})


def _base_detail() -> pd.DataFrame:
    kin = _build_current_kinetics()
    target = _load_target_map()
    floor = float(kin.params.b_overlap_floor)
    rows = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            raw_amp_ref = float(kin.hll_channel_amplitude(LAYER, REF_D, REF_ETA, T_COH, observable_mode=OBSERVABLE_MODE, N_max=N_MAX))
            amp_ref = float(kin._blend_observable_ref_amp(raw_amp_ref, LAYER, OBSERVABLE_MODE, float(D)))
            raw_amp = float(kin.hll_channel_amplitude(LAYER, float(D), float(eta), T_COH, observable_mode=OBSERVABLE_MODE, N_max=N_MAX))
            amp = float(kin._blend_observable_point_amp(raw_amp, LAYER, OBSERVABLE_MODE, float(D), float(eta)))
            raw_width = float(kin.hll_total_width_ratio_uv_rge(D=float(D), eta=float(eta), t_coh=T_COH, ref_D=REF_D, ref_eta=REF_ETA, N_max=N_MAX))
            width = float(kin._blend_observable_width_ratio(raw_width, float(D), float(eta)))
            refamp_target = kin._observable_ref_amp_anchor_target(LAYER, OBSERVABLE_MODE)
            row_mask = np.isclose(target['D'], float(D), rtol=0.0, atol=1e-9) & np.isclose(target['eta'], float(eta), rtol=0.0, atol=1e-9)
            trow = target.loc[row_mask].iloc[0]
            rows.append({
                'D': float(D),
                'eta': float(eta),
                'target_mu': float(trow['target_mu']),
                'target_chi2': float(trow['target_chi2']),
                'cur_amp': amp,
                'cur_amp_ref': amp_ref,
                'cur_width': width,
                'cur_mu': float((amp / max(amp_ref, 1e-30)) ** 2 / max(width, 1e-30)),
                'refamp_target': float(refamp_target) if refamp_target is not None else np.nan,
                'base_refamp_beta': float(kin._observable_ref_amp_anchor_effective_beta(float(D))),
                'floor': floor,
            })
    return pd.DataFrame(rows)


def _evaluate(base: pd.DataFrame, taper_peak: float, center_D: float, sigma_D: float) -> pd.DataFrame:
    df = base.copy()
    skew = np.array([_local_skew(d, center_D, sigma_D) for d in df['D']], dtype=float)
    df['refamp_beta_new'] = np.clip(df['base_refamp_beta'] + taper_peak * skew, 0.0, 1.0)
    df['amp_ref_new'] = [
        _geom_blend(ar, rt, beta, fl) if np.isfinite(rt) else ar
        for ar, rt, beta, fl in zip(df['cur_amp_ref'], df['refamp_target'], df['refamp_beta_new'], df['floor'])
    ]
    df['mu_new'] = (df['cur_amp'] / df['amp_ref_new'].clip(lower=1e-30)) ** 2 / df['cur_width'].clip(lower=1e-30)
    df['delta_mu'] = df['mu_new'] - df['target_mu']
    df['acceptance_mismatch'] = ((df['target_chi2'] <= 4.0) ^ ((((df['mu_new'] - 1.4) / 0.4) ** 2) <= 4.0)).astype(float)
    return df


def _summarize(df: pd.DataFrame, taper_peak: float, center_D: float, sigma_D: float) -> dict[str, float]:
    by_d = {float(d): float(np.max(np.abs(g['delta_mu']))) for d, g in df.groupby('D', sort=True)}
    hotspot = max(by_d.get(5.627118644067797, 0.0), by_d.get(5.898305084745763, 0.0))
    guard = max(by_d.get(6.169491525423729, 0.0), by_d.get(6.4406779661016955, 0.0), by_d.get(6.711864406779661, 0.0))
    return {
        'taper_peak': float(taper_peak),
        'center_D': float(center_D),
        'sigma_D': float(sigma_D),
        'p95_abs_delta_mu': float(np.percentile(np.abs(df['delta_mu']), 95.0)),
        'max_abs_delta_mu': float(np.max(np.abs(df['delta_mu']))),
        'acceptance_mismatch': float(np.mean(df['acceptance_mismatch'])),
        'D5p627_max_abs_delta_mu': float(by_d.get(5.627118644067797, np.nan)),
        'D5p898_max_abs_delta_mu': float(by_d.get(5.898305084745763, np.nan)),
        'D6p169_max_abs_delta_mu': float(by_d.get(6.169491525423729, np.nan)),
        'D6p441_max_abs_delta_mu': float(by_d.get(6.4406779661016955, np.nan)),
        'D6p712_max_abs_delta_mu': float(by_d.get(6.711864406779661, np.nan)),
        'joint_hotspot_objective': float(hotspot),
        'guard_objective': float(guard),
        'joint_objective': float(max(hotspot, 0.5 * guard)),
    }


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    sc = axes[0].scatter(summary['D5p627_max_abs_delta_mu'], summary['D5p898_max_abs_delta_mu'], c=summary['taper_peak'], cmap='coolwarm', s=30)
    axes[0].set_xlabel('D≈5.627 max |Δμ|')
    axes[0].set_ylabel('D≈5.898 max |Δμ|')
    axes[0].set_title('ref-amp skew hotspot tradeoff')
    fig.colorbar(sc, ax=axes[0], label='taper peak')
    axes[1].scatter(summary['guard_objective'], summary['joint_objective'], c=summary['sigma_D'], cmap='viridis', s=30)
    axes[1].set_xlabel('guard objective')
    axes[1].set_ylabel('joint objective')
    axes[1].set_title('guard vs hotspot')
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
    base = _base_detail()
    detail_rows=[]
    summary_rows=[]
    combos=[(p,c,s) for p in TAPER_PEAKS for c in CENTER_DS for s in SIGMA_DS]
    for i,(p,c,s) in enumerate(combos, start=1):
        print(f'[combo {i}/{len(combos)}] taper_peak={p:.2f} center_D={c:.2f} sigma_D={s:.2f}')
        df=_evaluate(base,p,c,s)
        summary_rows.append(_summarize(df,p,c,s))
        keep=df[['D','eta','delta_mu','acceptance_mismatch','refamp_beta_new']].copy()
        keep.insert(0,'sigma_D',float(s))
        keep.insert(0,'center_D',float(c))
        keep.insert(0,'taper_peak',float(p))
        detail_rows.append(keep)
    detail=pd.concat(detail_rows, ignore_index=True)
    summary=pd.DataFrame(summary_rows).sort_values(['joint_objective','guard_objective','p95_abs_delta_mu','center_D','sigma_D']).reset_index(drop=True)
    slices=summary[['taper_peak','center_D','sigma_D','D5p627_max_abs_delta_mu','D5p898_max_abs_delta_mu','D6p169_max_abs_delta_mu','D6p441_max_abs_delta_mu','D6p712_max_abs_delta_mu','joint_hotspot_objective','guard_objective','joint_objective']]

    detail_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_d56_refamp_skew_audit_detail.csv'
    summary_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_d56_refamp_skew_audit_summary.csv'
    slices_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_d56_refamp_skew_audit_slices.csv'
    png_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_d56_refamp_skew_audit.png'
    meta_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_d56_refamp_skew_audit_run_meta.json'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(json.dumps({'base_map': str(BASE_MAP), 'focus_D':[float(x) for x in FOCUS_D], 'eta_grid':[float(x) for x in ETA_GRID], 'taper_peaks':list(TAPER_PEAKS), 'center_Ds':list(CENTER_DS), 'sigma_Ds':list(SIGMA_DS)}, indent=2))
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.head(20).to_string(index=False))

if __name__ == '__main__':
    main()
