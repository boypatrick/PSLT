#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import types
from pathlib import Path

import numpy as np
import pandas as pd

from scan_hll_signal_strengths import PAPER_BASELINE, load_observations, make_baseline_kinetics

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'kinetic_action_chain'
PAPERDIR = ROOT / 'paper'
MAPDIR = ROOT / 'output' / 'hll_signal_strength'

CASES = [
    ('D21E21_holdout', 'D21 x E21 holdout', MAPDIR / 'hll_signal_strength_map_chain_mode_full_direct_D21E21_20260330_crossgrid_holdout_r1.csv'),
    ('D60E21_release', 'D60 x E21 release', MAPDIR / 'hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv'),
]
CASE_FILTER = {s.strip() for s in os.environ.get('CASE_FILTER', '').split(',') if s.strip()}
TARGETS = [4.0, 4.8, 6.4, 7.2, 8.0]
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
GNORM_BANDPASS = {
    'beta': float(os.environ.get('STRICT_GNORM_BETA', '0.25')),
    'center': float(os.environ.get('STRICT_GNORM_CENTER', '0.06')),
    'half_width': float(os.environ.get('STRICT_GNORM_HALF_WIDTH', '0.04')),
}
CASE_GNORM_BANDPASS = {
    'D21E21_holdout': {
        'beta': float(os.environ.get('STRICT_GNORM_BETA_D21', str(GNORM_BANDPASS['beta']))),
        'center': float(os.environ.get('STRICT_GNORM_CENTER_D21', str(GNORM_BANDPASS['center']))),
        'half_width': float(os.environ.get('STRICT_GNORM_HALF_WIDTH_D21', str(GNORM_BANDPASS['half_width']))),
    },
    'D60E21_release': {
        'beta': float(os.environ.get('STRICT_GNORM_BETA_D60', str(GNORM_BANDPASS['beta']))),
        'center': float(os.environ.get('STRICT_GNORM_CENTER_D60', str(GNORM_BANDPASS['center']))),
        'half_width': float(os.environ.get('STRICT_GNORM_HALF_WIDTH_D60', str(GNORM_BANDPASS['half_width']))),
    },
}
WIDTH_BANDS = [
    {'beta': 1.309, 'center': 4.0, 'half_width': 0.40},
    {'beta': float(os.environ.get('STRICT_D48_BETA', '0.75')), 'center': 4.8, 'half_width': 0.40},
    {'beta': float(os.environ.get('STRICT_D64_BETA', '0.0')), 'center': 6.4, 'half_width': 0.40},
    {'beta': float(os.environ.get('STRICT_D72_BETA', '0.35')), 'center': 7.2, 'half_width': 0.40},
    {'beta': 1.20, 'center': 8.0, 'half_width': 0.40},
]
CASE_D40_BETA = {
    'D21E21_holdout': float(os.environ.get('STRICT_D40_BETA_D21', '1.309')),
    'D60E21_release': float(os.environ.get('STRICT_D40_BETA_D60', '1.309')),
}
CASE_D48_BETA = {
    'D21E21_holdout': float(os.environ.get('STRICT_D48_BETA_D21', str(WIDTH_BANDS[1]['beta']))),
    'D60E21_release': float(os.environ.get('STRICT_D48_BETA_D60', str(WIDTH_BANDS[1]['beta']))),
}
CASE_D64_BETA = {
    'D21E21_holdout': float(os.environ.get('STRICT_D64_BETA_D21', str(WIDTH_BANDS[2]['beta']))),
    'D60E21_release': float(os.environ.get('STRICT_D64_BETA_D60', str(WIDTH_BANDS[2]['beta']))),
}
CASE_D72_BETA = {
    'D21E21_holdout': float(os.environ.get('STRICT_D72_BETA_D21', str(WIDTH_BANDS[3]['beta']))),
    'D60E21_release': float(os.environ.get('STRICT_D72_BETA_D60', str(WIDTH_BANDS[3]['beta']))),
}

TAG = os.environ.get('STRICT_GENERALIZATION_TAG', '').strip()
SUFFIX = f"_{TAG}" if TAG else ""
OUT_SUMMARY = OUTDIR / f'runtime_direct_detlin_generalization_fast_summary{SUFFIX}.csv'
OUT_DETAIL = OUTDIR / f'runtime_direct_detlin_generalization_fast_detail{SUFFIX}.csv'
OUT_DECISION = OUTDIR / f'runtime_direct_detlin_generalization_fast_decision{SUFFIX}.json'


def build_kinetics(case: str, d_min: float, d_max: float, d_num: int):
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
    for k, v in WIDTH_FIX.items():
        setattr(p, f'hll_match_{k}', float(v))
    p.hll_uv_runtime_direct_m2_cross_blend = 0.75

    gspec = dict(CASE_GNORM_BANDPASS.get(case, GNORM_BANDPASS))
    beta = float(gspec['beta'])
    center = float(gspec['center'])
    half_width = float(gspec['half_width'])
    def patched_gnorm(self, hll_diag, direct_diag, direct_b123):
        floor = float(self.params.hll_uv_coupling_floor)
        h = np.maximum(np.asarray(hll_diag, dtype=float).reshape(3), floor)
        d = np.maximum(np.asarray(direct_diag, dtype=float).reshape(3), floor)
        metric = float(max(min(np.log(h[0] / d[0]), np.log(h[2] / d[2])), 0.0))
        activation = float(max(0.0, 1.0 - abs(metric - center) / half_width))
        b = np.maximum(np.asarray(direct_b123, dtype=float).reshape(3), float(self.params.b_overlap_floor))
        b2_gate = float(np.clip((float(b[1]) - 0.75) / 0.10, 0.0, 1.0))
        return float(beta * activation * b2_gate)
    kin._runtime_direct_gnorm_blend_weight = types.MethodType(patched_gnorm, kin)

    width_bands = [dict(spec) for spec in WIDTH_BANDS]
    for spec in width_bands:
        if np.isclose(spec['center'], 4.0):
            spec['beta'] = float(CASE_D40_BETA.get(case, spec['beta']))
        if np.isclose(spec['center'], 4.8):
            spec['beta'] = float(CASE_D48_BETA.get(case, spec['beta']))
        if np.isclose(spec['center'], 6.4):
            spec['beta'] = float(CASE_D64_BETA.get(case, spec['beta']))
        if np.isclose(spec['center'], 7.2):
            spec['beta'] = float(CASE_D72_BETA.get(case, spec['beta']))

    original = kin._blend_observable_width_ratio
    def patched_width(self, width_ratio: float, D: float, eta: float) -> float:
        base = float(original(width_ratio=width_ratio, D=D, eta=eta))
        floor = float(self.params.b_overlap_floor)
        positive_log_width = float(max(np.log(max(base, floor)), 0.0))
        exponent = np.log(max(base, floor))
        for spec in width_bands:
            act = float(max(0.0, 1.0 - abs(float(D) - float(spec['center'])) / float(spec['half_width'])))
            exponent += float(spec['beta']) * act * positive_log_width
        return float(np.exp(exponent))
    kin._blend_observable_width_ratio = types.MethodType(patched_width, kin)
    return kin


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)
    obs = load_observations()['mumu']
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)

    detail_rows = []
    summary_rows = []
    cases = [spec for spec in CASES if not CASE_FILTER or spec[0] in CASE_FILTER]
    for case, label, path in cases:
        df = pd.read_csv(path)
        df['D'] = df['D'].astype(float)
        df['eta'] = df['eta'].astype(float)
        d_grid = sorted(df['D'].unique())
        slice_ds = sorted({float(d_grid[int(np.argmin(np.abs(np.asarray(d_grid) - t)))]) for t in TARGETS})
        kin = build_kinetics(case, min(d_grid), max(d_grid), len(d_grid))
        for D in slice_ds:
            sub = df[np.isclose(df['D'], D)]
            for row in sub.itertuples(index=False):
                mu = float(kin.hll_mu_pred(2, D=float(row.D), eta=float(row.eta), t_coh=T_COH, ref_D=REF_D, ref_eta=REF_ETA, observable_mode='eft_wilson_uv_rge', N_max=kin.params.hll_observable_nmax))
                chi2 = float(((mu - mu_obs) / sigma_obs) ** 2)
                detail_rows.append({
                    'case': case,
                    'case_label': label,
                    'target_D': float(min(TARGETS, key=lambda t: abs(t - float(row.D)))),
                    'eval_D': float(row.D),
                    'eta': float(row.eta),
                    'abs_delta_mu_mumu': float(abs(mu - float(row.mu_mumu))),
                    'acceptance_mismatch': float((float(row.chi2_mumu) <= 4.0) != (chi2 <= 4.0)),
                })
        cdf = pd.DataFrame([r for r in detail_rows if r['case'] == case])
        summary_rows.append({
            'case': case,
            'case_label': label,
            'n_points': int(len(cdf)),
            'acceptance_mismatch': float(cdf['acceptance_mismatch'].mean()),
            'p95_abs_delta_mu_mumu': float(np.percentile(cdf['abs_delta_mu_mumu'], 95.0)),
            'max_abs_delta_mu_mumu': float(cdf['abs_delta_mu_mumu'].max()),
        })

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(OUT_DETAIL, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    decision = {
        'all_cases_zero_mismatch': bool((summary['acceptance_mismatch'] == 0.0).all()),
        'worst_case_p95_abs_delta_mu_mumu': float(summary['p95_abs_delta_mu_mumu'].max()),
        'worst_case': str(summary.loc[summary['p95_abs_delta_mu_mumu'].idxmax(), 'case']),
    }
    OUT_DECISION.write_text(json.dumps(decision, indent=2))
    for path in [OUT_SUMMARY, OUT_DETAIL, OUT_DECISION]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())
    print(f'[saved] {OUT_SUMMARY}')
    print(json.dumps(decision, indent=2))

if __name__ == '__main__':
    main()
