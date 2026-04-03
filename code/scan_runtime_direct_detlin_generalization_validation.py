#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import types
from pathlib import Path

TMP_MPL = Path('/tmp/pslt_mplconfig')
TMP_MPL.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(TMP_MPL))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, load_observations, make_baseline_kinetics

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'kinetic_action_chain'
PAPERDIR = ROOT / 'paper'
MAPDIR = ROOT / 'output' / 'hll_signal_strength'

OUT_SUMMARY = OUTDIR / 'runtime_direct_detlin_generalization_validation_summary.csv'
OUT_DETAIL = OUTDIR / 'runtime_direct_detlin_generalization_validation_detail.csv'
OUT_DSUMMARY = OUTDIR / 'runtime_direct_detlin_generalization_validation_dsummary.csv'
OUT_PNG = OUTDIR / 'runtime_direct_detlin_generalization_validation.png'
OUT_META = OUTDIR / 'runtime_direct_detlin_generalization_validation_run_meta.json'
OUT_DECISION = OUTDIR / 'runtime_direct_detlin_generalization_validation_decision.json'

ALL_CASES = [
    {
        'case': 'D21E41_focus',
        'label': 'D21 x E41 focus',
        'map_path': MAPDIR / 'hll_signal_strength_map_chain_mode_full_direct_D21E41.csv',
    },
    {
        'case': 'D21E21_holdout',
        'label': 'D21 x E21 holdout',
        'map_path': MAPDIR / 'hll_signal_strength_map_chain_mode_full_direct_D21E21_20260330_crossgrid_holdout_r1.csv',
    },
    {
        'case': 'D60E21_release',
        'label': 'D60 x E21 release',
        'map_path': MAPDIR / 'hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv',
    },
    {
        'case': 'D60E41_holdout',
        'label': 'D60 x E41 holdout',
        'map_path': MAPDIR / 'hll_signal_strength_map_chain_mode_full_direct_D60E41_20260330_crossgrid_holdout_r1.csv',
    },
    {
        'case': 'D80E41_monitor',
        'label': 'D80 x E41 monitor',
        'map_path': MAPDIR / 'hll_signal_strength_map_chain_mode_full_direct_D80E41_20260330_d60_micro_monitor_r1.csv',
    },
    {
        'case': 'D100E41_monitor',
        'label': 'D100 x E41 monitor',
        'map_path': MAPDIR / 'hll_signal_strength_map_chain_mode_full_direct_D100E41_20260330_d60_micro_monitor_r1.csv',
    },
]



def _selected_cases() -> list[dict[str, object]]:
    raw = os.environ.get('STRICT_DIRECT_CASES', '').strip()
    if not raw:
        return list(ALL_CASES)
    wanted = {tok.strip() for tok in raw.split(',') if tok.strip()}
    return [case for case in ALL_CASES if str(case['case']) in wanted]

REF_D = float(PAPER_BASELINE['ref_D'])
REF_ETA = float(PAPER_BASELINE['ref_eta'])
T_COH = float(PAPER_BASELINE['t_coh'])

WIDTH_FIX = {
    'width_power_base': 0.80,
    'width_power_log_ratio_coeff': 0.15,
    'width_power_min': 0.60,
    'width_power_max': 0.85,
    'width_power_trigger_lo': 2.5,
    'width_power_trigger_hi': 5.0,
    'width_power_turnoff_lo': 10.0,
    'width_power_turnoff_hi': 20.0,
    'width_power_tail_logratio_lo': 1.5,
    'width_power_tail_logratio_hi': 2.5,
    'width_power_tail_reboost_max': 0.15,
}
GNORM_BANDPASS = {'beta': 0.25, 'center': 0.06, 'half_width': 0.04}
WIDTH_BANDS = [
    {'name': 'd4p0_width', 'beta': 1.309, 'center': 4.0, 'half_width': 0.40},
    {'name': 'd4p8_width', 'beta': 0.75, 'center': 4.8, 'half_width': 0.40},
    {'name': 'd7p2_width', 'beta': 0.35, 'center': 7.2, 'half_width': 0.40},
    {'name': 'd8p0_width', 'beta': 1.20, 'center': 8.0, 'half_width': 0.40},
]
FOCUS_SPECS = WIDTH_BANDS + [
    {'name': 'd6p4_gnorm', 'beta': GNORM_BANDPASS['beta'], 'center': 6.4, 'half_width': 0.40}
]


def _load_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['D'] = df['D'].astype(float)
    df['eta'] = df['eta'].astype(float)
    df['mu_mumu'] = df['mu_mumu'].astype(float)
    df['chi2_mumu'] = df['chi2_mumu'].astype(float)
    return df.sort_values(['D', 'eta']).reset_index(drop=True)


def _focus_mask(d_values: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(d_values, dtype=bool)
    for spec in FOCUS_SPECS:
        mask |= np.abs(d_values - float(spec['center'])) <= float(spec['half_width'])
    return mask


def _build_kinetics(d_min: float, d_max: float, d_num: int):
    kin = make_baseline_kinetics(
        observable_mode='eft_wilson_uv_rge',
        chain_mode='cell_direct_runtime_release_tailm2gnorm',
        d_min=float(d_min),
        d_max=float(d_max),
        d_num=int(d_num),
        uv_blend=0.0,
        uv_m2_power=1.0,
        uv_match_kappa_diag=0.0,
        uv_match_kappa_offdiag=0.0,
    )
    p = kin.params
    p.hll_match_width_power_base = float(WIDTH_FIX['width_power_base'])
    p.hll_match_width_power_log_ratio_coeff = float(WIDTH_FIX['width_power_log_ratio_coeff'])
    p.hll_match_width_power_min = float(WIDTH_FIX['width_power_min'])
    p.hll_match_width_power_max = float(WIDTH_FIX['width_power_max'])
    p.hll_match_width_power_trigger_lo = float(WIDTH_FIX['width_power_trigger_lo'])
    p.hll_match_width_power_trigger_hi = float(WIDTH_FIX['width_power_trigger_hi'])
    p.hll_match_width_power_turnoff_lo = float(WIDTH_FIX['width_power_turnoff_lo'])
    p.hll_match_width_power_turnoff_hi = float(WIDTH_FIX['width_power_turnoff_hi'])
    p.hll_match_width_power_tail_logratio_lo = float(WIDTH_FIX['width_power_tail_logratio_lo'])
    p.hll_match_width_power_tail_logratio_hi = float(WIDTH_FIX['width_power_tail_logratio_hi'])
    p.hll_match_width_power_tail_reboost_max = float(WIDTH_FIX['width_power_tail_reboost_max'])
    p.hll_uv_runtime_direct_m2_cross_blend = 0.75
    return kin


def _patch_gnorm_bandpass(kin) -> None:
    beta = float(GNORM_BANDPASS['beta'])
    center = float(GNORM_BANDPASS['center'])
    half_width = float(GNORM_BANDPASS['half_width'])

    def patched(self, hll_diag: np.ndarray, direct_diag: np.ndarray, direct_b123: np.ndarray) -> float:
        floor = float(self.params.hll_uv_coupling_floor)
        h = np.maximum(np.asarray(hll_diag, dtype=float).reshape(3), floor)
        d = np.maximum(np.asarray(direct_diag, dtype=float).reshape(3), floor)
        metric = float(max(min(np.log(h[0] / d[0]), np.log(h[2] / d[2])), 0.0))
        activation = float(max(0.0, 1.0 - abs(metric - center) / half_width))
        b = np.maximum(np.asarray(direct_b123, dtype=float).reshape(3), float(self.params.b_overlap_floor))
        b2_gate = float(np.clip((float(b[1]) - 0.75) / 0.10, 0.0, 1.0))
        return float(beta * activation * b2_gate)

    kin._runtime_direct_gnorm_blend_weight = types.MethodType(patched, kin)


def _patch_width_bands(kin) -> None:
    original = kin._blend_observable_width_ratio

    def patched(self, width_ratio: float, D: float, eta: float) -> float:
        base = float(original(width_ratio=width_ratio, D=D, eta=eta))
        floor = float(self.params.b_overlap_floor)
        positive_log_width = float(max(np.log(max(base, floor)), 0.0))
        exponent = np.log(max(base, floor))
        for spec in WIDTH_BANDS:
            act = float(max(0.0, 1.0 - abs(float(D) - float(spec['center'])) / float(spec['half_width'])))
            exponent += float(spec['beta']) * act * positive_log_width
        return float(np.exp(exponent))

    kin._blend_observable_width_ratio = types.MethodType(patched, kin)


def _evaluate_case(case: dict[str, object], mu_obs: float, sigma_obs: float) -> pd.DataFrame:
    df = _load_map(Path(case['map_path']))
    d_grid = sorted(df['D'].unique())
    eta_grid = sorted(df['eta'].unique())
    kin = _build_kinetics(min(d_grid), max(d_grid), len(d_grid))
    _patch_gnorm_bandpass(kin)
    _patch_width_bands(kin)

    rows = []
    for row in df.itertuples(index=False):
        mu = float(
            kin.hll_mu_pred(
                2,
                D=float(row.D),
                eta=float(row.eta),
                t_coh=T_COH,
                ref_D=REF_D,
                ref_eta=REF_ETA,
                observable_mode='eft_wilson_uv_rge',
                N_max=kin.params.hll_observable_nmax,
            )
        )
        chi2 = float(((mu - mu_obs) / sigma_obs) ** 2)
        rows.append(
            {
                'case': str(case['case']),
                'case_label': str(case['label']),
                'D': float(row.D),
                'eta': float(row.eta),
                'mu_candidate': mu,
                'mu_full': float(row.mu_mumu),
                'chi2_candidate': chi2,
                'chi2_full': float(row.chi2_mumu),
                'abs_delta_mu_mumu': float(abs(mu - float(row.mu_mumu))),
                'acceptance_mismatch': float((float(row.chi2_mumu) <= 4.0) != (chi2 <= 4.0)),
            }
        )

    out = pd.DataFrame(rows)
    mask = _focus_mask(out['D'].to_numpy(dtype=float))
    out['subset'] = np.where(mask, 'focus_band', 'complement')
    return out


def _subset_metrics(df: pd.DataFrame, subset_name: str) -> dict[str, float | str | int]:
    if df.empty:
        return {
            'subset': subset_name,
            'n_points': 0,
            'acceptance_mismatch': float('nan'),
            'p95_abs_delta_mu_mumu': float('nan'),
            'max_abs_delta_mu_mumu': float('nan'),
            'mean_abs_delta_mu_mumu': float('nan'),
            'worst_D': float('nan'),
        }
    per_d = df.groupby('D')['abs_delta_mu_mumu'].max().sort_values(ascending=False)
    return {
        'subset': subset_name,
        'n_points': int(len(df)),
        'acceptance_mismatch': float(df['acceptance_mismatch'].mean()),
        'p95_abs_delta_mu_mumu': float(np.percentile(df['abs_delta_mu_mumu'], 95.0)),
        'max_abs_delta_mu_mumu': float(df['abs_delta_mu_mumu'].max()),
        'mean_abs_delta_mu_mumu': float(df['abs_delta_mu_mumu'].mean()),
        'worst_D': float(per_d.index[0]),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)

    obs = load_observations()['mumu']
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)

    detail_frames = []
    summary_rows = []
    dsummary_rows = []
    selected_cases = _selected_cases()
    if not selected_cases:
        raise SystemExit('No matching cases selected')

    for case in selected_cases:
        detail = _evaluate_case(case, mu_obs=mu_obs, sigma_obs=sigma_obs)
        detail_frames.append(detail)

        all_metrics = _subset_metrics(detail, 'all_points')
        focus_metrics = _subset_metrics(detail[detail['subset'] == 'focus_band'], 'focus_band')
        comp_metrics = _subset_metrics(detail[detail['subset'] == 'complement'], 'complement')
        for metrics in [all_metrics, focus_metrics, comp_metrics]:
            summary_rows.append({'case': case['case'], 'case_label': case['label'], **metrics})

        per_d = detail.groupby('D').agg(
            n_points=('eta', 'size'),
            acceptance_mismatch=('acceptance_mismatch', 'mean'),
            p95_abs_delta_mu_mumu=('abs_delta_mu_mumu', lambda s: float(np.percentile(s, 95.0))),
            max_abs_delta_mu_mumu=('abs_delta_mu_mumu', 'max'),
            mean_abs_delta_mu_mumu=('abs_delta_mu_mumu', 'mean'),
        ).reset_index()
        per_d['case'] = case['case']
        per_d['case_label'] = case['label']
        dsummary_rows.append(per_d)

    detail_df = pd.concat(detail_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    dsummary_df = pd.concat(dsummary_rows, ignore_index=True)

    case_order = [case['case'] for case in selected_cases]
    label_map = {case['case']: case['label'] for case in selected_cases}
    summary_df['case'] = pd.Categorical(summary_df['case'], categories=case_order, ordered=True)
    summary_df = summary_df.sort_values(['case', 'subset']).reset_index(drop=True)

    all_df = summary_df[summary_df['subset'] == 'all_points'].copy()
    focus_df = summary_df[summary_df['subset'] == 'focus_band'].copy()
    comp_df = summary_df[summary_df['subset'] == 'complement'].copy()

    x = np.arange(len(case_order))
    labels = [label_map[c] for c in case_order]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].bar(x, all_df['p95_abs_delta_mu_mumu'], color='#4e79a7')
    axes[0, 0].axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    axes[0, 0].set_title('All-points p95 |Δmu_mumu|')
    axes[0, 0].set_ylabel('p95')
    axes[0, 0].set_xticks(x, labels, rotation=25, ha='right')
    axes[0, 0].grid(alpha=0.25, axis='y')

    axes[0, 1].bar(x - 0.18, focus_df['p95_abs_delta_mu_mumu'], width=0.36, label='focus band', color='#59a14f')
    axes[0, 1].bar(x + 0.18, comp_df['p95_abs_delta_mu_mumu'], width=0.36, label='complement', color='#f28e2b')
    axes[0, 1].axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    axes[0, 1].set_title('Subset p95 |Δmu_mumu|')
    axes[0, 1].set_ylabel('p95')
    axes[0, 1].set_xticks(x, labels, rotation=25, ha='right')
    axes[0, 1].grid(alpha=0.25, axis='y')
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(x - 0.18, all_df['acceptance_mismatch'], width=0.36, label='all points', color='#4e79a7')
    axes[1, 0].bar(x + 0.18, focus_df['acceptance_mismatch'], width=0.36, label='focus band', color='#59a14f')
    axes[1, 0].set_title('Acceptance mismatch')
    axes[1, 0].set_ylabel('fraction')
    axes[1, 0].set_xticks(x, labels, rotation=25, ha='right')
    axes[1, 0].grid(alpha=0.25, axis='y')
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(x - 0.18, focus_df['max_abs_delta_mu_mumu'], width=0.36, label='focus band', color='#59a14f')
    axes[1, 1].bar(x + 0.18, comp_df['max_abs_delta_mu_mumu'], width=0.36, label='complement', color='#f28e2b')
    axes[1, 1].axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    axes[1, 1].set_title('Subset max |Δmu_mumu|')
    axes[1, 1].set_ylabel('max')
    axes[1, 1].set_xticks(x, labels, rotation=25, ha='right')
    axes[1, 1].grid(alpha=0.25, axis='y')
    axes[1, 1].legend(fontsize=8)

    fig.suptitle('Strict all-direct broader-grid generalization validation')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    summary_df.to_csv(OUT_SUMMARY, index=False)
    detail_df.to_csv(OUT_DETAIL, index=False)
    dsummary_df.to_csv(OUT_DSUMMARY, index=False)

    decision = {
        'all_cases_focus_band_zero_mismatch': bool((focus_df['acceptance_mismatch'] == 0.0).all()),
        'all_cases_allpoints_zero_mismatch': bool((all_df['acceptance_mismatch'] == 0.0).all()),
        'worst_allpoints_p95_abs_delta_mu_mumu': float(all_df['p95_abs_delta_mu_mumu'].max()),
        'worst_complement_p95_abs_delta_mu_mumu': float(comp_df['p95_abs_delta_mu_mumu'].max()),
        'worst_allpoints_case': str(all_df.loc[all_df['p95_abs_delta_mu_mumu'].idxmax(), 'case']),
        'recommendation': 'focus_band_generalization_clean' if bool((all_df['acceptance_mismatch'] == 0.0).all()) and float(all_df['p95_abs_delta_mu_mumu'].max()) <= 1.0 else 'focus_band_closed_but_broader_grid_still_monitor',
    }
    OUT_META.write_text(json.dumps({
        'cases': selected_cases,
        'gnorm_bandpass': GNORM_BANDPASS,
        'width_bands': WIDTH_BANDS,
        'width_fix': WIDTH_FIX,
    }, indent=2))
    OUT_DECISION.write_text(json.dumps(decision, indent=2))

    for path in [OUT_SUMMARY, OUT_DETAIL, OUT_DSUMMARY, OUT_PNG, OUT_META, OUT_DECISION]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())

    print(f'[saved] {OUT_SUMMARY}')
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
