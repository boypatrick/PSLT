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
OBSERVABLE_MODE = 'eft_wilson_uv_rge'
T_COH = float(PAPER_BASELINE['t_coh'])
N_MAX = int(PAPER_BASELINE['hll_observable_nmax'])
LAYER = 2
REF_D = 9.966101694915254
REF_ETA = 1.0
D_GRID60 = np.linspace(4.0, 20.0, 60)
ETA_GRID = np.linspace(0.2, 4.0, 21)
HOT_TARGETS = [4.814, 5.627, 6.712]
GUARD_TARGETS = [5.898, 6.169, 6.441]
HOT_D = [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - t)))]) for t in HOT_TARGETS]
GUARD_D = [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - t)))]) for t in GUARD_TARGETS]
ALPHAS = [0.75, 0.80, 0.825, 0.85]
TAPERS = [0.0, 0.01, 0.02]
BOOST_PEAKS = [0.0, 0.3, 0.5]
BOOST_CENTERS = [4.813559322033898, 5.9, 6.711864406779661]
BOOST_SIGMA = 0.20


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


def _build_ref():
    return make_baseline_kinetics(chain_mode='full_direct', **_common_kwargs())


def _build_cur(alpha: float, taper: float, boost_peak: float, boost_center: float):
    return make_baseline_kinetics(
        chain_mode='cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost',
        **_common_kwargs(),
        observable_width_anchor_blend_override=float(alpha),
        observable_width_anchor_blend_taper_override=float(taper),
        observable_width_anchor_blend_taper_center_D_override=6.0,
        observable_width_anchor_blend_taper_sigma_D_override=0.4,
        observable_width_anchor_boost_peak_override=float(boost_peak),
        observable_width_anchor_boost_center_D_override=float(boost_center),
        observable_width_anchor_boost_sigma_D_override=float(BOOST_SIGMA),
    )


def _mu(kin, D: float, eta: float) -> float:
    return float(
        kin.hll_mu_pred(
            LAYER,
            D=D,
            eta=eta,
            t_coh=T_COH,
            ref_D=REF_D,
            ref_eta=REF_ETA,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )


def _slice_max(kin_ref, kin_cur, Ds: list[float]) -> dict[float, float]:
    out = {}
    for D in Ds:
        vals = []
        for eta in ETA_GRID:
            vals.append(abs(_mu(kin_cur, D, float(eta)) - _mu(kin_ref, D, float(eta))))
        out[float(D)] = float(max(vals))
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    kin_ref = _build_ref()
    rows = []
    total = len(ALPHAS) * len(TAPERS) * len(BOOST_PEAKS) * len(BOOST_CENTERS)
    idx = 0
    for alpha in ALPHAS:
        for taper in TAPERS:
            for boost_peak in BOOST_PEAKS:
                for boost_center in BOOST_CENTERS:
                    idx += 1
                    print(f'[combo {idx}/{total}] alpha={alpha:.3f} taper={taper:.3f} peak={boost_peak:.2f} center={boost_center:.3f}', flush=True)
                    kin_cur = _build_cur(alpha, taper, boost_peak, boost_center)
                    hot = _slice_max(kin_ref, kin_cur, HOT_D)
                    guard = _slice_max(kin_ref, kin_cur, GUARD_D)
                    rows.append({
                        'alpha': float(alpha),
                        'taper': float(taper),
                        'boost_peak': float(boost_peak),
                        'boost_center_D': float(boost_center),
                        'boost_sigma_D': float(BOOST_SIGMA),
                        'D4p814_max_abs_delta_mu': float(hot[HOT_D[0]]),
                        'D5p627_max_abs_delta_mu': float(hot[HOT_D[1]]),
                        'D6p712_max_abs_delta_mu': float(hot[HOT_D[2]]),
                        'D5p898_guard_max_abs_delta_mu': float(guard[GUARD_D[0]]),
                        'D6p169_guard_max_abs_delta_mu': float(guard[GUARD_D[1]]),
                        'D6p441_guard_max_abs_delta_mu': float(guard[GUARD_D[2]]),
                        'hotspot_objective': float(max(hot.values())),
                        'guard_objective': float(max(guard.values())),
                        'joint_objective': float(max(max(hot.values()), max(guard.values()))),
                    })
    df = pd.DataFrame(rows).sort_values(['joint_objective','hotspot_objective','D5p627_max_abs_delta_mu']).reset_index(drop=True)
    detail_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_d60_widthband_audit_detail.csv'
    summary_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_d60_widthband_audit_summary.csv'
    slices_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_d60_widthband_audit_slices.csv'
    png_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_d60_widthband_audit.png'
    meta_path = OUTDIR / 'model_chain_fullwidthrefamp_pointamp2_widthboost_d60_widthband_audit_run_meta.json'
    df.to_csv(detail_path, index=False)
    df.to_csv(summary_path, index=False)
    df.to_csv(slices_path, index=False)
    top = df.head(12)
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.plot(range(len(top)), top['D4p814_max_abs_delta_mu'], marker='o', label='D≈4.814')
    ax.plot(range(len(top)), top['D5p627_max_abs_delta_mu'], marker='o', label='D≈5.627')
    ax.plot(range(len(top)), top['D6p712_max_abs_delta_mu'], marker='o', label='D≈6.712')
    ax.plot(range(len(top)), top['D6p169_guard_max_abs_delta_mu'], marker='o', linestyle='--', label='guard D≈6.169')
    ax.plot(range(len(top)), top['D6p441_guard_max_abs_delta_mu'], marker='o', linestyle='--', label='guard D≈6.441')
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f"{r.alpha:.3f}\n{r.taper:.3f}\n{r.boost_peak:.2f}\n{r.boost_center_D:.2f}" for r in top.itertuples()], fontsize=7)
    ax.set_ylabel('max |Δμ|')
    ax.set_title('D60 reviewer width-band candidates')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    meta_path.write_text(json.dumps({'hot_D': HOT_D, 'guard_D': GUARD_D, 'alphas': ALPHAS, 'tapers': TAPERS, 'boost_peaks': BOOST_PEAKS, 'boost_centers': BOOST_CENTERS, 'boost_sigma': BOOST_SIGMA}, indent=2))
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())
    print(df.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
