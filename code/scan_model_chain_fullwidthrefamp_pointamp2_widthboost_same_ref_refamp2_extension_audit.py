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

PEAK2S = (0.25, 0.5, 0.75, 1.0)
CENTER2S = (5.95, 6.03, 6.10)
SIGMA2S = (0.08, 0.12, 0.16)


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


def _build_base_detail() -> pd.DataFrame:
    kin_full, kin_cur = _build_kinetics()
    rows: list[dict[str, float]] = []
    for D in FOCUS_D:
        for eta in ETA_GRID:
            target_mu = _mu(kin_full, float(D), float(eta))
            raw_amp_ref = float(kin_cur.hll_channel_amplitude(LAYER, REF_D, REF_ETA, T_COH, observable_mode=OBSERVABLE_MODE, N_max=N_MAX))
            amp_ref = float(kin_cur._blend_observable_ref_amp(raw_amp_ref, LAYER, OBSERVABLE_MODE, float(D)))
            raw_amp = float(kin_cur.hll_channel_amplitude(LAYER, float(D), float(eta), T_COH, observable_mode=OBSERVABLE_MODE, N_max=N_MAX))
            amp = float(kin_cur._blend_observable_point_amp(raw_amp, LAYER, OBSERVABLE_MODE, float(D), float(eta)))
            raw_width = float(kin_cur.hll_total_width_ratio_uv_rge(D=float(D), eta=float(eta), t_coh=T_COH, ref_D=REF_D, ref_eta=REF_ETA, N_max=N_MAX))
            width = float(kin_cur._blend_observable_width_ratio(raw_width, float(D), float(eta)))
            refamp_target = kin_cur._observable_ref_amp_anchor_target(LAYER, OBSERVABLE_MODE)
            rows.append({
                'D': float(D),
                'eta': float(eta),
                'target_mu': target_mu,
                'cur_amp': amp,
                'cur_amp_ref': amp_ref,
                'cur_width': width,
                'refamp_target': float(refamp_target) if refamp_target is not None else np.nan,
                'floor': float(kin_cur.params.b_overlap_floor),
            })
    return pd.DataFrame(rows)


def _evaluate_combo(base: pd.DataFrame, peak2: float, center2: float, sigma2: float) -> pd.DataFrame:
    df = base.copy()
    df['extra_refamp2_beta'] = [_local_beta(peak2, center2, sigma2, d) for d in df['D']]
    df['amp_ref_new'] = [
        _geom_blend(ar, rt, br, fl) if np.isfinite(rt) else ar
        for ar, rt, br, fl in zip(df['cur_amp_ref'], df['refamp_target'], df['extra_refamp2_beta'], df['floor'])
    ]
    df['mu_new'] = (df['cur_amp'] / df['amp_ref_new'].clip(lower=1e-30)) ** 2 / df['cur_width'].clip(lower=1e-30)
    df['delta_mu'] = df['mu_new'] - df['target_mu']
    return df


def _summarize_combo(df: pd.DataFrame, peak2: float, center2: float, sigma2: float) -> dict[str, float]:
    by_d = {float(d): float(np.max(np.abs(g['delta_mu']))) for d, g in df.groupby('D', sort=True)}
    hotspot = max(by_d.get(FOCUS_D[0], 0.0), by_d.get(FOCUS_D[1], 0.0))
    guard = max(by_d.get(FOCUS_D[2], 0.0), by_d.get(FOCUS_D[3], 0.0))
    return {
        'peak2': float(peak2),
        'center2_D': float(center2),
        'sigma2_D': float(sigma2),
        'p95_abs_delta_mu': float(np.percentile(np.abs(df['delta_mu']), 95.0)),
        'max_abs_delta_mu': float(np.max(np.abs(df['delta_mu']))),
        'D5p898_max_abs_delta_mu': float(by_d.get(FOCUS_D[0], np.nan)),
        'D6p169_max_abs_delta_mu': float(by_d.get(FOCUS_D[1], np.nan)),
        'D6p441_max_abs_delta_mu': float(by_d.get(FOCUS_D[2], np.nan)),
        'D6p712_max_abs_delta_mu': float(by_d.get(FOCUS_D[3], np.nan)),
        'hotspot_objective': float(hotspot),
        'guard_objective': float(guard),
        'joint_objective': float(max(hotspot, 0.5 * guard)),
    }


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    sc = axes[0].scatter(summary['D5p898_max_abs_delta_mu'], summary['D6p169_max_abs_delta_mu'], c=summary['peak2'], cmap='viridis', s=36)
    axes[0].set_xlabel('D≈5.898 max |Δμ|')
    axes[0].set_ylabel('D≈6.169 max |Δμ|')
    axes[0].set_title('same-ref refamp2 extension')
    fig.colorbar(sc, ax=axes[0], label='peak2')
    sc2 = axes[1].scatter(summary['guard_objective'], summary['joint_objective'], c=summary['center2_D'], cmap='plasma', s=36)
    axes[1].set_xlabel('guard objective')
    axes[1].set_ylabel('joint objective')
    axes[1].set_title('guard vs hotspot')
    fig.colorbar(sc2, ax=axes[1], label='center2_D')
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
    combos = [(p, c, s) for p in PEAK2S for c in CENTER2S for s in SIGMA2S]
    for i, (peak2, center2, sigma2) in enumerate(combos, start=1):
        print(f'[combo {i}/{len(combos)}] peak2={peak2:.2f} center2={center2:.2f} sigma2={sigma2:.2f}')
        df = _evaluate_combo(base, peak2, center2, sigma2)
        keep = df[['D', 'eta', 'target_mu', 'mu_new', 'delta_mu', 'extra_refamp2_beta']].copy()
        keep.insert(0, 'sigma2_D', float(sigma2))
        keep.insert(0, 'center2_D', float(center2))
        keep.insert(0, 'peak2', float(peak2))
        detail_rows.append(keep)
        summary_rows.append(_summarize_combo(df, peak2, center2, sigma2))
    detail = pd.concat(detail_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(['joint_objective', 'guard_objective', 'p95_abs_delta_mu', 'center2_D', 'sigma2_D']).reset_index(drop=True)
    slices = summary[['peak2','center2_D','sigma2_D','D5p898_max_abs_delta_mu','D6p169_max_abs_delta_mu','D6p441_max_abs_delta_mu','D6p712_max_abs_delta_mu','hotspot_objective','guard_objective','joint_objective']]

    detail_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_same_ref_refamp2_extension_audit_detail.csv'
    summary_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_same_ref_refamp2_extension_audit_summary.csv'
    slices_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_same_ref_refamp2_extension_audit_slices.csv'
    png_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_same_ref_refamp2_extension_audit.png'
    meta_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_same_ref_refamp2_extension_audit_run_meta.json'

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(json.dumps({
        'current_mode': CURRENT_MODE,
        'focus_D': [float(x) for x in FOCUS_D],
        'eta_grid': [float(x) for x in ETA_GRID],
        'ref_D': float(REF_D),
        'peak2s': list(PEAK2S),
        'center2s': list(CENTER2S),
        'sigma2s': list(SIGMA2S),
    }, indent=2))
    _mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
