#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'chi_open_system'

SCAN_CONFIG = {
    'family': {
        'path': OUTDIR / 'chi_open_system_parent_bath_family_scan.csv',
        'params': ['m', 'u', 'v'],
    },
    'generator_affinity': {
        'path': OUTDIR / 'chi_open_system_parent_bath_generator_affinity_scan.csv',
        'params': ['q_ss', 'q_bb', 'q_sb'],
    },
    'log_coordinate': {
        'path': OUTDIR / 'chi_open_system_parent_bath_log_coordinate_scan.csv',
        'params': ['p_sys', 'p_spec'],
    },
    'normal_coordinate': {
        'path': OUTDIR / 'chi_open_system_parent_bath_normal_coordinate_scan.csv',
        'params': ['zeta_sys', 'zeta_spec'],
    },
}

P95_COLS = [
    'p95_block11_rel_residual_identifiable',
    'p95_block22_rel_residual',
    'p95_trace_rel_residual',
    'p95_det_rel_residual_identifiable',
    'p95_anisotropy_abs_residual_identifiable',
]
MAX_COLS = [
    'max_block11_rel_residual_identifiable',
    'max_block22_rel_residual',
    'max_trace_rel_residual',
    'max_det_rel_residual_identifiable',
    'max_anisotropy_abs_residual_identifiable',
]


def is_canonical(df: pd.DataFrame, params: list[str]) -> pd.Series:
    mask = np.ones(len(df), dtype=bool)
    for p in params:
        mask &= np.isclose(df[p].to_numpy(dtype=float), 0.0)
    return pd.Series(mask, index=df.index)


def lorentz_factor(r: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + r * r)


def evaluate_gap_floor(g_floor: float, r_on_shell: np.ndarray, delta_q_ref: float) -> dict[str, float]:
    sep = lorentz_factor(r_on_shell / max(g_floor, 1e-30)) - lorentz_factor(r_on_shell)
    sep_min = float(np.min(sep))
    eps_factor_max = float(np.max((1.0 + r_on_shell * r_on_shell) / (g_floor * g_floor + r_on_shell * r_on_shell)))
    if delta_q_ref < sep_min:
        tail_upper = float(delta_q_ref / sep_min)
        eps0_upper = float(delta_q_ref / (sep_min - delta_q_ref))
        eps_delta_upper = float(eps0_upper * eps_factor_max)
    else:
        tail_upper = np.nan
        eps0_upper = np.nan
        eps_delta_upper = np.nan
    return {
        'separator_min': sep_min,
        'tail_mass_upper': tail_upper,
        'eps0_upper': eps0_upper,
        'eps_delta_upper': eps_delta_upper,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Aggregate the four audited deformation scans into a family-free operator-witness transfer audit.'
    )
    ap.add_argument('--objective-tol', type=float, default=1e-6)
    ap.add_argument('--safety-factor', type=float, default=0.5)
    ap.add_argument('--neighborhood-summary', default=str(OUTDIR / 'chi_open_system_operator_detdeficit_neighborhood_bound_summary.csv'))
    ap.add_argument('--deltaq-summary', default=str(OUTDIR / 'chi_open_system_deltaq_source_summary.csv'))
    ap.add_argument('--micro-input', default=str(OUTDIR / 'chi_open_system_micro_bridge_map.csv'))
    ap.add_argument('--detail-output', default=str(OUTDIR / 'chi_open_system_operator_witness_transfer_detail.csv'))
    ap.add_argument('--summary-output', default=str(OUTDIR / 'chi_open_system_operator_witness_transfer_summary.csv'))
    args = ap.parse_args()

    rows = []
    canonical_rows = []
    noncanonical_rows = []
    for name, cfg in SCAN_CONFIG.items():
        df = pd.read_csv(cfg['path']).copy()
        canon_mask = is_canonical(df, cfg['params'])
        df['audit'] = name
        df['canonical'] = canon_mask.to_numpy(dtype=bool)
        for p in cfg['params']:
            df[f'param_{p}'] = df[p]
        df['p95_linf'] = np.max(df[P95_COLS].to_numpy(dtype=float), axis=1)
        df['p95_l2'] = np.sqrt(np.sum(np.square(df[P95_COLS].to_numpy(dtype=float)), axis=1))
        df['max_linf'] = np.max(df[MAX_COLS].to_numpy(dtype=float), axis=1)
        df['max_l2'] = np.sqrt(np.sum(np.square(df[MAX_COLS].to_numpy(dtype=float)), axis=1))
        rows.append(df)
        canonical_rows.append(df[df['canonical']].copy())
        noncanonical_rows.append(df[~df['canonical']].copy())

    all_df = pd.concat(rows, ignore_index=True)
    canonical_df = pd.concat(canonical_rows, ignore_index=True)
    noncanonical_df = pd.concat(noncanonical_rows, ignore_index=True)

    nearest_p95 = noncanonical_df.sort_values(['p95_linf', 'objective']).iloc[0]
    nearest_max = noncanonical_df.sort_values(['max_linf', 'objective']).iloc[0]
    nearest_all = noncanonical_df.sort_values(['objective', 'p95_linf']).iloc[0]

    p95_safe_radius = float(args.safety_factor * nearest_p95['p95_linf'])
    max_safe_radius = float(args.safety_factor * nearest_max['max_linf'])

    canonical_p95_sup = float(canonical_df['p95_linf'].max())
    canonical_max_sup = float(canonical_df['max_linf'].max())
    canonical_objective = float(canonical_df['objective'].max())

    detail = noncanonical_df[[
        'audit', 'objective', 'p95_linf', 'p95_l2', 'max_linf', 'max_l2', *P95_COLS, *MAX_COLS,
        'param_m', 'param_u', 'param_v', 'param_q_ss', 'param_q_bb', 'param_q_sb',
        'param_p_sys', 'param_p_spec', 'param_zeta_sys', 'param_zeta_spec'
    ]].sort_values(['p95_linf', 'objective']).reset_index(drop=True)

    neigh_df = pd.read_csv(args.neighborhood_summary)
    neigh = neigh_df.iloc[(neigh_df['objective_tol'] - args.objective_tol).abs().idxmin()]
    deltaq = pd.read_csv(args.deltaq_summary).iloc[0]
    micro = pd.read_csv(args.micro_input)
    r_on_shell = (micro['delta_micro'] / np.maximum(micro['omega1'], 1e-30)).to_numpy(dtype=float)
    g_lift = float(neigh['lifted_g_floor'])
    lift_eval = evaluate_gap_floor(g_lift, r_on_shell, float(deltaq['max_abs_delta_q_parent_log_vs_gap']))

    summary = pd.DataFrame([
        {
            'objective_tol': float(args.objective_tol),
            'safety_factor': float(args.safety_factor),
            'n_scans': int(len(SCAN_CONFIG)),
            'canonical_objective_max': canonical_objective,
            'canonical_p95_linf_max': canonical_p95_sup,
            'canonical_max_linf_max': canonical_max_sup,
            'nearest_noncanonical_objective': float(nearest_all['objective']),
            'nearest_noncanonical_p95_linf': float(nearest_p95['p95_linf']),
            'nearest_noncanonical_max_linf': float(nearest_max['max_linf']),
            'p95_safe_radius': p95_safe_radius,
            'max_safe_radius': max_safe_radius,
            'p95_safe_over_canonical': p95_safe_radius / max(canonical_p95_sup, 1e-30),
            'max_safe_over_canonical': max_safe_radius / max(canonical_max_sup, 1e-30),
            'nearest_p95_audit': str(nearest_p95['audit']),
            'nearest_max_audit': str(nearest_max['audit']),
            'nearest_p95_objective': float(nearest_p95['objective']),
            'nearest_max_objective': float(nearest_max['objective']),
            'u_neighborhood_upper': float(neigh['u_neighborhood_upper']),
            'lifted_g_floor': g_lift,
            'heff_loewner_factor': float(neigh['heff_loewner_factor']),
            'separator_min_from_lifted_g': lift_eval['separator_min'],
            'actual_delta_q_max': float(deltaq['max_abs_delta_q_parent_log_vs_gap']),
            'tail_mass_upper_from_actual_delta_q': lift_eval['tail_mass_upper'],
            'eps0_upper_from_actual_delta_q': lift_eval['eps0_upper'],
            'eps_delta_upper_from_actual_delta_q': lift_eval['eps_delta_upper'],
        }
    ])

    for out in [args.detail_output, args.summary_output]:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print(args.detail_output)
    print(args.summary_output)


if __name__ == '__main__':
    main()
