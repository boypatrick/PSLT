#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from scan_hll_signal_strengths import PAPER_BASELINE, load_observations, make_baseline_kinetics

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'kinetic_action_chain'
PAPERDIR = ROOT / 'paper'
MAP = ROOT / 'output' / 'hll_signal_strength' / 'hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv'

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

D40_BETA = 1.304
D48_BETA = 0.65
D72_BETA = 0.35
D80_BETA = 1.20
D21_G = (0.30, 0.06, 0.04)
CURRENT = (0.243, 0.049, 0.023)
BOUNDS = {
    'g_beta': (0.238, 0.248),
    'g_center': (0.046, 0.052),
    'g_half_width': (0.020, 0.028),
}
N_SAMPLES = 16
OUT_TAG = 'd60_d64_sobol_doe_v1'
OUT_SAMPLES = OUTDIR / f'runtime_direct_detlin_{OUT_TAG}_samples.csv'
OUT_EFFECTS = OUTDIR / f'runtime_direct_detlin_{OUT_TAG}_effects.csv'
OUT_DECISION = OUTDIR / f'runtime_direct_detlin_{OUT_TAG}_summary.json'


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
    for k, v in WIDTH_FIX.items():
        setattr(p, f'hll_match_{k}', float(v))
    p.hll_uv_runtime_direct_m2_cross_blend = 0.75
    return kin


def _patch_width_bands(kin):
    import types

    original = kin._blend_observable_width_ratio
    bands = [
        {'beta': D40_BETA, 'center': 4.0, 'half_width': 0.40},
        {'beta': D48_BETA, 'center': 4.8, 'half_width': 0.40},
        {'beta': D72_BETA, 'center': 7.2, 'half_width': 0.40},
        {'beta': D80_BETA, 'center': 8.0, 'half_width': 0.40},
    ]

    def patched(self, width_ratio: float, D: float, eta: float) -> float:
        base = float(original(width_ratio=width_ratio, D=D, eta=eta))
        floor = float(self.params.b_overlap_floor)
        positive_log_width = float(max(np.log(max(base, floor)), 0.0))
        exponent = np.log(max(base, floor))
        for spec in bands:
            act = float(max(0.0, 1.0 - abs(float(D) - float(spec['center'])) / float(spec['half_width'])))
            exponent += float(spec['beta']) * act * positive_log_width
        return float(np.exp(exponent))

    kin._blend_observable_width_ratio = types.MethodType(patched, kin)


def _patch_gnorm_caseaware(kin, g_beta: float, g_center: float, g_half_width: float):
    import types

    def patched(self, hll_diag: np.ndarray, direct_diag: np.ndarray, direct_b123: np.ndarray) -> float:
        floor = float(self.params.hll_uv_coupling_floor)
        h = np.maximum(np.asarray(hll_diag, dtype=float).reshape(3), floor)
        d = np.maximum(np.asarray(direct_diag, dtype=float).reshape(3), floor)
        metric = float(max(min(np.log(h[0] / d[0]), np.log(h[2] / d[2])), 0.0))
        activation = float(max(0.0, 1.0 - abs(metric - g_center) / g_half_width))
        b = np.maximum(np.asarray(direct_b123, dtype=float).reshape(3), float(self.params.b_overlap_floor))
        b2_gate = float(np.clip((float(b[1]) - 0.75) / 0.10, 0.0, 1.0))
        return float(g_beta * activation * b2_gate)

    kin._runtime_direct_gnorm_blend_weight = types.MethodType(patched, kin)


def _design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    x1 = df['g_beta'].to_numpy()
    x2 = df['g_center'].to_numpy()
    x3 = df['g_half_width'].to_numpy()
    cols = [
        np.ones(len(df)), x1, x2, x3,
        x1 * x1, x2 * x2, x3 * x3,
        x1 * x2, x1 * x3, x2 * x3,
    ]
    names = ['const', 'g_beta', 'g_center', 'g_half_width', 'g_beta2', 'g_center2', 'g_half_width2', 'g_beta:g_center', 'g_beta:g_half_width', 'g_center:g_half_width']
    return np.column_stack(cols), names


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPERDIR.mkdir(parents=True, exist_ok=True)
    obs = load_observations()['mumu']
    mu_obs = float(obs.mu_obs)
    sigma_obs = max(float(obs.sigma_obs), 1e-12)

    df = pd.read_csv(MAP)
    df['D'] = df['D'].astype(float)
    df['eta'] = df['eta'].astype(float)
    target_D = float(sorted(df['D'].unique())[int(np.argmin(np.abs(np.array(sorted(df['D'].unique())) - 6.4)))])
    sub = df[np.isclose(df['D'], target_D)].copy()

    d_grid = sorted(df['D'].unique())
    kin_base = _build_kinetics(min(d_grid), max(d_grid), len(d_grid))
    _patch_width_bands(kin_base)

    sampler = qmc.Sobol(d=3, scramble=True, seed=64)
    unit = sampler.random_base2(m=4)
    lowers = np.array([BOUNDS['g_beta'][0], BOUNDS['g_center'][0], BOUNDS['g_half_width'][0]])
    uppers = np.array([BOUNDS['g_beta'][1], BOUNDS['g_center'][1], BOUNDS['g_half_width'][1]])
    pts = qmc.scale(unit, lowers, uppers)

    rows = []
    for i, (g_beta, g_center, g_half_width) in enumerate(pts, start=1):
        kin = _build_kinetics(min(d_grid), max(d_grid), len(d_grid))
        _patch_width_bands(kin)
        _patch_gnorm_caseaware(kin, float(g_beta), float(g_center), float(g_half_width))
        detail_rows = []
        for row in sub.itertuples(index=False):
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
            detail_rows.append({
                'abs_delta_mu_mumu': float(abs(mu - float(row.mu_mumu))),
                'acceptance_mismatch': float((float(row.chi2_mumu) <= 4.0) != (chi2 <= 4.0)),
            })
        dd = pd.DataFrame(detail_rows)
        rows.append({
            'sample_id': i,
            'g_beta': float(g_beta),
            'g_center': float(g_center),
            'g_half_width': float(g_half_width),
            'p95_abs_delta_mu_mumu': float(np.percentile(dd['abs_delta_mu_mumu'], 95.0)),
            'max_abs_delta_mu_mumu': float(dd['abs_delta_mu_mumu'].max()),
            'mismatch': float(dd['acceptance_mismatch'].mean()),
            'score': float(np.percentile(dd['abs_delta_mu_mumu'], 95.0) + 25.0 * dd['acceptance_mismatch'].mean()),
        })
        partial = pd.DataFrame(rows).sort_values('score')
        partial.to_csv(OUT_SAMPLES, index=False)
        print(f"[progress] {i}/{len(pts)} beta={g_beta:.6f} center={g_center:.6f} width={g_half_width:.6f} p95={rows[-1]['p95_abs_delta_mu_mumu']:.6f} mismatch={rows[-1]['mismatch']:.6f}", flush=True)

    samples = pd.DataFrame(rows).sort_values('score').reset_index(drop=True)
    X, names = _design_matrix(samples)
    y = samples['p95_abs_delta_mu_mumu'].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    effects = pd.DataFrame({'term': names, 'coef': coef})
    nonconst = effects[effects['term'] != 'const'].copy()
    nonconst['abs_coef'] = nonconst['coef'].abs()
    nonconst = nonconst.sort_values('abs_coef', ascending=False)
    effects.to_csv(OUT_EFFECTS, index=False)

    # Predict optimum from surrogate over a dense random candidate cloud inside the same box.
    rng = np.random.default_rng(64064)
    cloud = np.column_stack([
        rng.uniform(*BOUNDS['g_beta'], size=4000),
        rng.uniform(*BOUNDS['g_center'], size=4000),
        rng.uniform(*BOUNDS['g_half_width'], size=4000),
    ])
    cloud_df = pd.DataFrame(cloud, columns=['g_beta', 'g_center', 'g_half_width'])
    Xc, _ = _design_matrix(cloud_df)
    cloud_df['pred_p95'] = Xc @ coef
    best_pred = cloud_df.sort_values('pred_p95').iloc[0].to_dict()
    best_obs = samples.iloc[0].to_dict()

    summary = {
        'current_reference': {
            'g_beta': CURRENT[0],
            'g_center': CURRENT[1],
            'g_half_width': CURRENT[2],
        },
        'bounds': BOUNDS,
        'n_samples': len(samples),
        'best_observed': best_obs,
        'top_effects': nonconst.head(6).to_dict(orient='records'),
        'predicted_next_best': best_pred,
        'recommendation': 'validate_predicted_next_best_exactly_if_predicted_gain_exceeds_noise',
    }
    OUT_DECISION.write_text(json.dumps(summary, indent=2))

    for path in [OUT_SAMPLES, OUT_EFFECTS, OUT_DECISION]:
        (PAPERDIR / path.name).write_bytes(path.read_bytes())

    print(f'[saved] {OUT_SAMPLES}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
