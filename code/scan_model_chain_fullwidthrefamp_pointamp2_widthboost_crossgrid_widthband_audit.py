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

REF_D60 = 9.966101694915254
REF_D21 = 9.6
REF_ETA = 1.0

D_GRID60 = np.linspace(4.0, 20.0, 60)
ETA_GRID60 = np.linspace(0.2, 4.0, 21)
D_GRID21 = np.linspace(4.0, 20.0, 21)
ETA_GRID21 = np.linspace(0.2, 4.0, 41)

HOT60_TARGETS = [4.814, 5.627, 6.712]
GUARD60_TARGETS = [6.169, 6.441]
GUARD21_TARGETS = [4.0, 6.4, 8.0]

HOT60_D = [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - t)))]) for t in HOT60_TARGETS]
GUARD60_D = [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - t)))]) for t in GUARD60_TARGETS]
GUARD21_D = [float(D_GRID21[int(np.argmin(np.abs(D_GRID21 - t)))]) for t in GUARD21_TARGETS]

ALPHAS = [0.76, 0.77, 0.78, 0.79, 0.80]
TAPERS = [0.0125, 0.015, 0.0175]
BOOST_PEAKS = [0.0, 0.15, 0.30]
BOOST_SIGMAS = [0.10, 0.15, 0.20]
BOOST_CENTER = 5.9


def _common_kwargs(d_num: int) -> dict[str, float | int | str | bool]:
    return dict(
        observable_mode=OBSERVABLE_MODE,
        d_min=4.0,
        d_max=20.0,
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


def _build_ref(d_num: int):
    return make_baseline_kinetics(chain_mode='full_direct', **_common_kwargs(d_num))


def _build_cur(d_num: int, alpha: float, taper: float, boost_peak: float, boost_sigma: float):
    return make_baseline_kinetics(
        chain_mode='cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost',
        **_common_kwargs(d_num),
        observable_width_anchor_blend_override=float(alpha),
        observable_width_anchor_blend_taper_override=float(taper),
        observable_width_anchor_blend_taper_center_D_override=6.0,
        observable_width_anchor_blend_taper_sigma_D_override=0.4,
        observable_width_anchor_boost_peak_override=float(boost_peak),
        observable_width_anchor_boost_center_D_override=float(BOOST_CENTER),
        observable_width_anchor_boost_sigma_D_override=float(boost_sigma),
    )


def _mu(kin, D: float, eta: float, ref_D: float) -> float:
    return float(
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


def _slice_max(kin_ref, kin_cur, Ds: list[float], etas: np.ndarray, ref_D: float) -> dict[float, float]:
    out: dict[float, float] = {}
    for D in Ds:
        vals = []
        for eta in etas:
            vals.append(abs(_mu(kin_cur, D, float(eta), ref_D) - _mu(kin_ref, D, float(eta), ref_D)))
        out[float(D)] = float(max(vals))
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    kin_ref60 = _build_ref(60)
    kin_ref21 = _build_ref(21)
    rows = []
    total = len(ALPHAS) * len(TAPERS) * len(BOOST_PEAKS) * len(BOOST_SIGMAS)
    idx = 0
    for alpha in ALPHAS:
        for taper in TAPERS:
            for boost_peak in BOOST_PEAKS:
                for boost_sigma in BOOST_SIGMAS:
                    idx += 1
                    print(
                        f'[combo {idx}/{total}] alpha={alpha:.3f} taper={taper:.4f} peak={boost_peak:.2f} sigma={boost_sigma:.2f}',
                        flush=True,
                    )
                    kin_cur60 = _build_cur(60, alpha, taper, boost_peak, boost_sigma)
                    kin_cur21 = _build_cur(21, alpha, taper, boost_peak, boost_sigma)
                    hot60 = _slice_max(kin_ref60, kin_cur60, HOT60_D, ETA_GRID60, REF_D60)
                    guard60 = _slice_max(kin_ref60, kin_cur60, GUARD60_D, ETA_GRID60, REF_D60)
                    guard21 = _slice_max(kin_ref21, kin_cur21, GUARD21_D, ETA_GRID21, REF_D21)
                    rows.append({
                        'alpha': float(alpha),
                        'taper': float(taper),
                        'boost_peak': float(boost_peak),
                        'boost_center_D': float(BOOST_CENTER),
                        'boost_sigma_D': float(boost_sigma),
                        'D60_D4p814_max_abs_delta_mu': float(hot60[HOT60_D[0]]),
                        'D60_D5p627_max_abs_delta_mu': float(hot60[HOT60_D[1]]),
                        'D60_D6p712_max_abs_delta_mu': float(hot60[HOT60_D[2]]),
                        'D60_guard_D6p169_max_abs_delta_mu': float(guard60[GUARD60_D[0]]),
                        'D60_guard_D6p441_max_abs_delta_mu': float(guard60[GUARD60_D[1]]),
                        'D21_guard_D4p0_max_abs_delta_mu': float(guard21[GUARD21_D[0]]),
                        'D21_guard_D6p4_max_abs_delta_mu': float(guard21[GUARD21_D[1]]),
                        'D21_guard_D8p0_max_abs_delta_mu': float(guard21[GUARD21_D[2]]),
                        'hotspot_objective': float(max(hot60.values())),
                        'guard60_objective': float(max(guard60.values())),
                        'guard21_objective': float(max(guard21.values())),
                        'joint_objective': float(max(max(hot60.values()), max(guard60.values()), max(guard21.values()))),
                    })
    df = pd.DataFrame(rows).sort_values(
        ['joint_objective', 'guard21_objective', 'guard60_objective', 'hotspot_objective', 'D60_D5p627_max_abs_delta_mu']
    ).reset_index(drop=True)
    stem = 'model_chain_fullwidthrefamp_pointamp2_widthboost_crossgrid_widthband_audit'
    detail_path = OUTDIR / f'{stem}_detail.csv'
    summary_path = OUTDIR / f'{stem}_summary.csv'
    slices_path = OUTDIR / f'{stem}_slices.csv'
    png_path = OUTDIR / f'{stem}.png'
    meta_path = OUTDIR / f'{stem}_run_meta.json'
    df.to_csv(detail_path, index=False)
    df.to_csv(summary_path, index=False)
    df.to_csv(slices_path, index=False)
    top = df.head(15)
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    ax.plot(range(len(top)), top['D60_D5p627_max_abs_delta_mu'], marker='o', label='D60 D≈5.627')
    ax.plot(range(len(top)), top['D60_guard_D6p169_max_abs_delta_mu'], marker='o', linestyle='--', label='D60 guard D≈6.169')
    ax.plot(range(len(top)), top['D60_guard_D6p441_max_abs_delta_mu'], marker='o', linestyle='--', label='D60 guard D≈6.441')
    ax.plot(range(len(top)), top['D21_guard_D6p4_max_abs_delta_mu'], marker='o', linestyle=':', label='D21 guard D=6.4')
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f"{r.alpha:.3f}\n{r.taper:.4f}\n{r.boost_peak:.2f}\n{r.boost_sigma_D:.2f}" for r in top.itertuples()], fontsize=7)
    ax.set_ylabel('max |Δμ|')
    ax.set_title('Cross-grid D60/D21 width-band candidates')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    meta_path.write_text(json.dumps({
        'hot60_D': HOT60_D,
        'guard60_D': GUARD60_D,
        'guard21_D': GUARD21_D,
        'alphas': ALPHAS,
        'tapers': TAPERS,
        'boost_peaks': BOOST_PEAKS,
        'boost_center_D': BOOST_CENTER,
        'boost_sigmas': BOOST_SIGMAS,
    }, indent=2))
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())
    print(df.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
