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
REF_CAND = 9.694915254237287
REF_TUNED = 9.966101694915254
REF_SCAN = np.array([
    9.423728813559322,
    9.694915254237287,
    9.966101694915254,
    10.23728813559322,
], dtype=float)
REF_ETA = 1.0
LAYER = 2
OBSERVABLE_MODE = 'eft_wilson_uv_rge'
T_COH = float(PAPER_BASELINE['t_coh'])
N_MAX = int(PAPER_BASELINE['hll_observable_nmax'])
CURRENT_MODE = 'cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost'


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


def _mu(kin, D: float, eta: float, ref_D: float) -> float:
    return float(
        kin.hll_mu_pred(
            LAYER,
            D,
            eta,
            T_COH,
            ref_D=ref_D,
            ref_eta=REF_ETA,
            observable_mode=OBSERVABLE_MODE,
            N_max=N_MAX,
        )
    )


def _build_detail() -> pd.DataFrame:
    full, cur = _build_kinetics()
    rows: list[dict[str, float | str]] = []
    for ref_D in REF_SCAN:
        print(f'[ref {ref_D:.12f}] evaluating focus grid')
        for D in FOCUS_D:
            for eta in ETA_GRID:
                mu_full = _mu(full, float(D), float(eta), float(ref_D))
                mu_cur = _mu(cur, float(D), float(eta), float(ref_D))
                delta = mu_cur - mu_full
                rows.append({
                    'ref_D': float(ref_D),
                    'ref_bucket': (
                        'candidate_ref' if np.isclose(ref_D, REF_CAND) else
                        'reviewer_ref' if np.isclose(ref_D, REF_TUNED) else
                        'neighbor_ref'
                    ),
                    'D': float(D),
                    'eta': float(eta),
                    'mu_full': float(mu_full),
                    'mu_current': float(mu_cur),
                    'delta_mu': float(delta),
                    'abs_delta_mu': float(abs(delta)),
                })
    return pd.DataFrame(rows)


def _summaries(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    slice_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    for ref_D, ref_df in detail.groupby('ref_D', sort=True):
        by_d = {}
        for D, g in ref_df.groupby('D', sort=True):
            g = g.sort_values('eta')
            mean = float(g['delta_mu'].mean())
            std = float(g['delta_mu'].std(ddof=1)) if len(g) > 1 else 0.0
            p95 = float(np.percentile(np.abs(g['delta_mu']), 95.0))
            max_abs = float(np.max(np.abs(g['delta_mu'])))
            rel_std = float(std / max(abs(mean), 1e-12))
            by_d[float(D)] = p95
            slice_rows.append({
                'ref_D': float(ref_D),
                'D': float(D),
                'mean_delta_mu': mean,
                'std_delta_mu': std,
                'rel_std_delta_mu': rel_std,
                'p95_abs_delta_mu': p95,
                'max_abs_delta_mu': max_abs,
                'eta_flat_offset_score': float(max_abs - p95),
            })
        summary_rows.append({
            'ref_D': float(ref_D),
            'D5p627_p95_abs_delta_mu': by_d.get(5.627118644067797, np.nan),
            'D5p898_p95_abs_delta_mu': by_d.get(5.898305084745763, np.nan),
            'D6p169_p95_abs_delta_mu': by_d.get(6.169491525423729, np.nan),
            'D6p441_p95_abs_delta_mu': by_d.get(6.4406779661016955, np.nan),
            'D6p712_p95_abs_delta_mu': by_d.get(6.711864406779661, np.nan),
            'anchor_sensitive_objective': float(by_d.get(5.627118644067797, np.nan)),
            'intrinsic_objective': float(max(by_d.get(5.898305084745763, 0.0), by_d.get(6.169491525423729, 0.0))),
        })
    slices = pd.DataFrame(slice_rows).sort_values(['ref_D', 'D']).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values('ref_D').reset_index(drop=True)
    return summary, slices


def _plot(summary: pd.DataFrame, slices: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axes[0].plot(summary['ref_D'], summary['D5p627_p95_abs_delta_mu'], marker='o', label='D≈5.627')
    axes[0].plot(summary['ref_D'], summary['D5p898_p95_abs_delta_mu'], marker='o', label='D≈5.898')
    axes[0].plot(summary['ref_D'], summary['D6p169_p95_abs_delta_mu'], marker='o', label='D≈6.169')
    axes[0].set_xlabel('ref_D')
    axes[0].set_ylabel('p95 |Δμ|')
    axes[0].set_title('Reference sensitivity')
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    pivot = slices.pivot(index='D', columns='ref_D', values='rel_std_delta_mu')
    im = axes[1].imshow(pivot.values, aspect='auto', cmap='magma_r')
    axes[1].set_yticks(np.arange(len(pivot.index)), [f'{d:.3f}' for d in pivot.index])
    axes[1].set_xticks(np.arange(len(pivot.columns)), [f'{d:.3f}' for d in pivot.columns], rotation=25)
    axes[1].set_xlabel('ref_D')
    axes[1].set_ylabel('D')
    axes[1].set_title('eta-flatness (rel std of Δμ)')
    fig.colorbar(im, ax=axes[1], label='rel std')
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _write_mirror(paths: list[Path]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for src in paths:
        (PAPER_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail = _build_detail()
    summary, slices = _summaries(detail)

    detail_path = OUTDIR / 'model_chain_d60_reviewer_anchor_alignment_audit_detail.csv'
    summary_path = OUTDIR / 'model_chain_d60_reviewer_anchor_alignment_audit_summary.csv'
    slices_path = OUTDIR / 'model_chain_d60_reviewer_anchor_alignment_audit_slices.csv'
    png_path = OUTDIR / 'model_chain_d60_reviewer_anchor_alignment_audit.png'
    meta_path = OUTDIR / 'model_chain_d60_reviewer_anchor_alignment_audit_run_meta.json'

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, slices, png_path)
    meta_path.write_text(json.dumps({
        'current_mode': CURRENT_MODE,
        'ref_scan': [float(x) for x in REF_SCAN],
        'focus_D': [float(x) for x in FOCUS_D],
        'eta_grid': [float(x) for x in ETA_GRID],
        'observable_mode': OBSERVABLE_MODE,
    }, indent=2))
    _write_mirror([detail_path, summary_path, slices_path, png_path, meta_path])
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
