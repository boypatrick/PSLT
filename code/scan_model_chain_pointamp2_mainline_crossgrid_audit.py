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
REF_ETA = 1.0
REF_D60 = 9.966101694915254
REF_D21 = 9.6
D_GRID60 = np.linspace(4.0, 20.0, 60)
ETA_GRID60 = np.linspace(0.2, 4.0, 21)
D_GRID21 = np.linspace(4.0, 20.0, 21)
ETA_GRID21 = np.linspace(0.2, 4.0, 41)
HOT60_TARGETS = [6.169, 6.441]
HOT21_TARGETS = [6.4]
GUARD60_TARGETS = [5.627, 6.712]
GUARD21_TARGETS = [4.0, 8.0]
HOT60_D = [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - t)))]) for t in HOT60_TARGETS]
HOT21_D = [float(D_GRID21[int(np.argmin(np.abs(D_GRID21 - t)))]) for t in HOT21_TARGETS]
GUARD60_D = [float(D_GRID60[int(np.argmin(np.abs(D_GRID60 - t)))]) for t in GUARD60_TARGETS]
GUARD21_D = [float(D_GRID21[int(np.argmin(np.abs(D_GRID21 - t)))]) for t in GUARD21_TARGETS]
PEAK2S = [0.55, 0.65]
CENTER2S = [6.63, 6.71]
SIGMA2S = [0.08, 0.10]
PEAK3S = [0.0, 0.25]
CENTER3S = [6.24, 6.31]
SIGMA3S = [0.06, 0.10]


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


def _build_cur(d_num: int, peak2: float, center2: float, sigma2: float, peak3: float, center3: float, sigma3: float):
    return make_baseline_kinetics(
        chain_mode='cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost',
        **_common_kwargs(d_num),
        observable_point_amp_anchor_peak2_override=float(peak2),
        observable_point_amp_anchor_center_D2_override=float(center2),
        observable_point_amp_anchor_sigma_D2_override=float(sigma2),
        observable_point_amp_anchor_peak3_override=float(peak3),
        observable_point_amp_anchor_center_D3_override=float(center3),
        observable_point_amp_anchor_sigma_D3_override=float(sigma3),
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
    total = len(PEAK2S) * len(CENTER2S) * len(SIGMA2S) * len(PEAK3S) * len(CENTER3S) * len(SIGMA3S)
    idx = 0
    for peak2 in PEAK2S:
        for center2 in CENTER2S:
            for sigma2 in SIGMA2S:
                for peak3 in PEAK3S:
                    for center3 in CENTER3S:
                        for sigma3 in SIGMA3S:
                            idx += 1
                            print(f'[combo {idx}/{total}] p2={peak2:.2f} c2={center2:.2f} s2={sigma2:.2f} p3={peak3:.2f} c3={center3:.2f} s3={sigma3:.2f}', flush=True)
                            kin_cur60 = _build_cur(60, peak2, center2, sigma2, peak3, center3, sigma3)
                            kin_cur21 = _build_cur(21, peak2, center2, sigma2, peak3, center3, sigma3)
                            hot60 = _slice_max(kin_ref60, kin_cur60, HOT60_D, ETA_GRID60, REF_D60)
                            hot21 = _slice_max(kin_ref21, kin_cur21, HOT21_D, ETA_GRID21, REF_D21)
                            guard60 = _slice_max(kin_ref60, kin_cur60, GUARD60_D, ETA_GRID60, REF_D60)
                            guard21 = _slice_max(kin_ref21, kin_cur21, GUARD21_D, ETA_GRID21, REF_D21)
                            rows.append({
                                'peak2': float(peak2),
                                'center2_D': float(center2),
                                'sigma2_D': float(sigma2),
                                'peak3': float(peak3),
                                'center3_D': float(center3),
                                'sigma3_D': float(sigma3),
                                'D60_D6p169_max_abs_delta_mu': float(hot60[HOT60_D[0]]),
                                'D60_D6p441_max_abs_delta_mu': float(hot60[HOT60_D[1]]),
                                'D21_D6p4_max_abs_delta_mu': float(hot21[HOT21_D[0]]),
                                'D60_guard_D5p627_max_abs_delta_mu': float(guard60[GUARD60_D[0]]),
                                'D60_guard_D6p712_max_abs_delta_mu': float(guard60[GUARD60_D[1]]),
                                'D21_guard_D4p0_max_abs_delta_mu': float(guard21[GUARD21_D[0]]),
                                'D21_guard_D8p0_max_abs_delta_mu': float(guard21[GUARD21_D[1]]),
                                'hotspot_objective': float(max(max(hot60.values()), max(hot21.values()))),
                                'guard_objective': float(max(max(guard60.values()), max(guard21.values()))),
                                'joint_objective': float(max(max(hot60.values()), max(hot21.values()), max(guard60.values()), max(guard21.values()))),
                            })
    df = pd.DataFrame(rows).sort_values(
        ['joint_objective', 'hotspot_objective', 'guard_objective', 'D21_D6p4_max_abs_delta_mu', 'D60_D6p441_max_abs_delta_mu']
    ).reset_index(drop=True)
    stem = 'model_chain_pointamp2_mainline_crossgrid_audit'
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
    ax.plot(range(len(top)), top['D60_D6p169_max_abs_delta_mu'], marker='o', label='D60 D≈6.169')
    ax.plot(range(len(top)), top['D60_D6p441_max_abs_delta_mu'], marker='o', label='D60 D≈6.441')
    ax.plot(range(len(top)), top['D21_D6p4_max_abs_delta_mu'], marker='o', linestyle='--', label='D21 D=6.4')
    ax.plot(range(len(top)), top['D60_guard_D5p627_max_abs_delta_mu'], marker='o', linestyle=':', label='D60 guard D≈5.627')
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([f"{r.peak2:.2f}\n{r.center2_D:.2f}\n{r.sigma2_D:.2f}\n{r.peak3:.2f}\n{r.center3_D:.2f}\n{r.sigma3_D:.2f}" for r in top.itertuples()], fontsize=6)
    ax.set_ylabel('max |Δμ|')
    ax.set_title('Cross-grid point-amp mainline candidates')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    meta_path.write_text(json.dumps({
        'hot60_D': HOT60_D,
        'hot21_D': HOT21_D,
        'guard60_D': GUARD60_D,
        'guard21_D': GUARD21_D,
        'peak2s': PEAK2S,
        'center2s': CENTER2S,
        'sigma2s': SIGMA2S,
        'peak3s': PEAK3S,
        'center3s': CENTER3S,
        'sigma3s': SIGMA3S,
    }, indent=2))
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())
    print(df.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
