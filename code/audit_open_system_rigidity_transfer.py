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


def is_canonical(df: pd.DataFrame, params: list[str]) -> pd.Series:
    mask = np.ones(len(df), dtype=bool)
    for p in params:
        mask &= np.isclose(df[p].to_numpy(dtype=float), 0.0)
    return pd.Series(mask, index=df.index)


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Audit canonical-rigidity transfer at fixed objective tolerance across the audited parent-bath deformation families.'
    )
    ap.add_argument('--objective-tol', type=float, default=1e-6)
    ap.add_argument('--detail-output', default=str(OUTDIR / 'chi_open_system_rigidity_transfer_detail.csv'))
    ap.add_argument('--summary-output', default=str(OUTDIR / 'chi_open_system_rigidity_transfer_summary.csv'))
    args = ap.parse_args()

    rows = []
    for name, cfg in SCAN_CONFIG.items():
        df = pd.read_csv(cfg['path']).copy()
        if 'objective' not in df.columns:
            raise KeyError(f"Missing objective column in {cfg['path']}")
        canon_mask = is_canonical(df, cfg['params'])
        admissible = df[df['objective'] <= args.objective_tol].copy()
        canonical_rows = df[canon_mask].copy()
        canonical_obj = float(canonical_rows['objective'].min())
        noncanon = df[~canon_mask].copy()
        min_noncanon = float(noncanon['objective'].min())
        admissible_noncanon = admissible[~is_canonical(admissible, cfg['params'])].copy()
        row = {
            'audit': name,
            'objective_tol': float(args.objective_tol),
            'canonical_objective': canonical_obj,
            'min_noncanonical_objective': min_noncanon,
            'selection_gap_to_noncanonical': min_noncanon - canonical_obj,
            'admissible_points_total': int(len(admissible)),
            'admissible_points_canonical': int(canon_mask[df['objective'] <= args.objective_tol].sum()),
            'admissible_points_noncanonical': int(len(admissible_noncanon)),
            'canonical_is_unique_admissible': int(len(admissible) == 1 and len(admissible_noncanon) == 0),
            'selection_gap_over_tol': (min_noncanon - canonical_obj) / max(float(args.objective_tol), 1e-30),
            'selection_gap_over_canonical': (min_noncanon - canonical_obj) / max(canonical_obj, 1e-30),
        }
        for p in cfg['params']:
            row[f'canonical_{p}'] = 0.0
        if len(admissible_noncanon) > 0:
            best = admissible_noncanon.sort_values('objective').iloc[0]
            for p in cfg['params']:
                row[f'first_noncanonical_{p}'] = float(best[p])
            row['first_noncanonical_objective'] = float(best['objective'])
        else:
            best = noncanon.sort_values('objective').iloc[0]
            for p in cfg['params']:
                row[f'first_noncanonical_{p}'] = float(best[p])
            row['first_noncanonical_objective'] = float(best['objective'])
        rows.append(row)

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {
            'objective_tol': float(args.objective_tol),
            'n_audits': int(len(detail)),
            'all_canonical_unique_admissible': int(detail['canonical_is_unique_admissible'].min() == 1),
            'max_canonical_objective': float(detail['canonical_objective'].max()),
            'min_noncanonical_objective': float(detail['min_noncanonical_objective'].min()),
            'min_selection_gap_to_noncanonical': float(detail['selection_gap_to_noncanonical'].min()),
            'min_selection_gap_over_tol': float(detail['selection_gap_over_tol'].min()),
            'min_selection_gap_over_canonical': float(detail['selection_gap_over_canonical'].min()),
            'total_admissible_points': int(detail['admissible_points_total'].sum()),
            'total_noncanonical_admissible_points': int(detail['admissible_points_noncanonical'].sum()),
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
