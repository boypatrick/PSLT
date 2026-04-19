#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'chi_open_system'


def q(series: pd.Series, p: float) -> float:
    arr = series.to_numpy(dtype=float)
    return float(np.quantile(arr, p)) if arr.size else float('nan')


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Derive the audited operator-witness chart radius directly from the exported exact statement/cocycle/potential maps.'
    )
    ap.add_argument('--statement-map', default=str(OUTDIR / 'chi_open_system_parent_bath_statement_map.csv'))
    ap.add_argument('--cocycle-summary', default=str(OUTDIR / 'chi_open_system_parent_bath_cocycle_summary.csv'))
    ap.add_argument('--potential-summary', default=str(OUTDIR / 'chi_open_system_parent_bath_potential_summary.csv'))
    ap.add_argument('--witness-transfer-summary', default=str(OUTDIR / 'chi_open_system_operator_witness_transfer_summary.csv'))
    ap.add_argument('--detail-output', default=str(OUTDIR / 'chi_open_system_exact_operator_chart_source_detail.csv'))
    ap.add_argument('--summary-output', default=str(OUTDIR / 'chi_open_system_exact_operator_chart_source_summary.csv'))
    args = ap.parse_args()

    stmt = pd.read_csv(args.statement_map).copy()
    phi_ident = stmt['bath_phi_identifiable'].to_numpy(dtype=bool)
    if not np.any(phi_ident):
        raise RuntimeError('No phi-identifiable rows in statement map.')

    detail = pd.DataFrame({
        'D': stmt['D'].to_numpy(dtype=float),
        'phi_identifiable': phi_ident.astype(int),
        'block11_residual_abs': np.where(phi_ident, stmt['bath_norm_phi_residual_abs'].to_numpy(dtype=float), np.nan),
        'block22_residual_abs': stmt['bath_norm_mix_residual_abs'].to_numpy(dtype=float),
        'trace_residual_abs': stmt['parent_bath_trace_residual_abs'].to_numpy(dtype=float),
        'det_residual_abs': np.where(phi_ident, stmt['parent_bath_det_residual_abs'].to_numpy(dtype=float), np.nan),
        'anisotropy_residual_abs': np.where(phi_ident, stmt['parent_log_rate_anisotropy_residual_abs'].to_numpy(dtype=float), np.nan),
    })
    detail['witness_linf'] = np.nanmax(detail[[
        'block11_residual_abs',
        'block22_residual_abs',
        'trace_residual_abs',
        'det_residual_abs',
        'anisotropy_residual_abs',
    ]].to_numpy(dtype=float), axis=1)

    p95_block11 = q(stmt.loc[phi_ident, 'bath_norm_phi_residual_abs'], 0.95)
    p95_block22 = q(stmt['bath_norm_mix_residual_abs'], 0.95)
    p95_trace = q(stmt['parent_bath_trace_residual_abs'], 0.95)
    p95_det = q(stmt.loc[phi_ident, 'parent_bath_det_residual_abs'], 0.95)
    p95_aniso = q(stmt.loc[phi_ident, 'parent_log_rate_anisotropy_residual_abs'], 0.95)

    max_block11 = float(stmt.loc[phi_ident, 'bath_norm_phi_residual_abs'].max())
    max_block22 = float(stmt['bath_norm_mix_residual_abs'].max())
    max_trace = float(stmt['parent_bath_trace_residual_abs'].max())
    max_det = float(stmt.loc[phi_ident, 'parent_bath_det_residual_abs'].max())
    max_aniso = float(stmt.loc[phi_ident, 'parent_log_rate_anisotropy_residual_abs'].max())

    p95_linf = max(p95_block11, p95_block22, p95_trace, p95_det, p95_aniso)
    max_linf = max(max_block11, max_block22, max_trace, max_det, max_aniso)

    cocycle = pd.read_csv(args.cocycle_summary).iloc[0]
    potential = pd.read_csv(args.potential_summary).iloc[0]
    witness = pd.read_csv(args.witness_transfer_summary).iloc[0]

    summary = pd.DataFrame([
        {
            'n_rows': int(len(stmt)),
            'D_min': float(stmt['D'].min()),
            'D_max': float(stmt['D'].max()),
            'exact_operator_p95_block11': p95_block11,
            'exact_operator_p95_block22': p95_block22,
            'exact_operator_p95_trace': p95_trace,
            'exact_operator_p95_det': p95_det,
            'exact_operator_p95_anisotropy': p95_aniso,
            'exact_operator_max_block11': max_block11,
            'exact_operator_max_block22': max_block22,
            'exact_operator_max_trace': max_trace,
            'exact_operator_max_det': max_det,
            'exact_operator_max_anisotropy': max_aniso,
            'exact_operator_p95_linf': p95_linf,
            'exact_operator_max_linf': max_linf,
            'p95_safe_radius': float(witness['p95_safe_radius']),
            'max_safe_radius': float(witness['max_safe_radius']),
            'p95_safe_over_exact_operator': float(witness['p95_safe_radius']) / max(p95_linf, 1e-30),
            'max_safe_over_exact_operator': float(witness['max_safe_radius']) / max(max_linf, 1e-30),
            'max_abs_triangle_defect_mix': float(cocycle['max_abs_triangle_defect_mix']),
            'max_abs_chain_recovery_residual_mix': float(potential['max_abs_chain_recovery_residual_mix']),
            'lifted_g_floor': float(witness['lifted_g_floor']),
            'heff_loewner_factor': float(witness['heff_loewner_factor']),
            'tail_mass_upper_from_actual_delta_q': float(witness['tail_mass_upper_from_actual_delta_q']),
            'eps_delta_upper_from_actual_delta_q': float(witness['eps_delta_upper_from_actual_delta_q']),
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
