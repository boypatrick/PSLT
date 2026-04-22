#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    moment = pd.read_csv(OUT_DIR / 'chi_open_system_parity_contrast_diag_anchor_moment_source_detail.csv')
    row = moment[moment['D'] == 11.0].iloc[0]

    singleton = pd.read_csv(OUT_DIR / 'chi_open_system_exact_schur_singleton_static_anchor_source_summary.csv').iloc[0]
    req_global = float(singleton['required_diag_floor_global'])
    req_local = float(singleton['required_diag_floor_local'])

    P = float(row['P_diag_anchor'])
    N = float(row['N_diag_anchor'])
    I = float(row['I_diag_anchor'])
    nu = float(row['neg_over_pos'])

    req_P_global = req_global / max(1.0 - nu, 1e-300)
    req_P_local = req_local / max(1.0 - nu, 1e-300)
    req_nu_global = 1.0 - req_global / max(P, 1e-300)
    req_nu_local = 1.0 - req_local / max(P, 1e-300)

    summary = {
        'D': 11.0,
        'I_diag_anchor_11': I,
        'P_diag_11': P,
        'N_diag_11': N,
        'nu_11': nu,
        'split_reconstruction_residual': abs(I - (P - N)),
        'required_diag_floor_global': req_global,
        'required_diag_floor_local': req_local,
        'required_P_diag_global_given_nu11': req_P_global,
        'required_P_diag_local_given_nu11': req_P_local,
        'required_nu11_global_given_Pdiag': req_nu_global,
        'required_nu11_local_given_Pdiag': req_nu_local,
        'Pdiag_over_required_global': P / max(req_P_global, 1e-300),
        'Pdiag_over_required_local': P / max(req_P_local, 1e-300),
        'nu_slack_global': req_nu_global - nu,
        'nu_slack_local': req_nu_local - nu,
        'preferred_parent_side_route': 'P_diag_lower_bound',
    }

    detail_rows = [
        {'term': 'I_diag_anchor_11', 'value': I, 'role': 'target'},
        {'term': 'P_diag_11', 'value': P, 'role': 'positive_moment'},
        {'term': 'N_diag_11', 'value': N, 'role': 'adverse_moment'},
        {'term': 'nu_11', 'value': nu, 'role': 'local_cancellation_ratio'},
        {'term': 'required_diag_floor_global', 'value': req_global, 'role': 'global_target'},
        {'term': 'required_diag_floor_local', 'value': req_local, 'role': 'local_target'},
        {'term': 'required_P_diag_global_given_nu11', 'value': req_P_global, 'role': 'global_positive_moment_target'},
        {'term': 'required_P_diag_local_given_nu11', 'value': req_P_local, 'role': 'local_positive_moment_target'},
        {'term': 'required_nu11_global_given_Pdiag', 'value': req_nu_global, 'role': 'global_ratio_target'},
        {'term': 'required_nu11_local_given_Pdiag', 'value': req_nu_local, 'role': 'local_ratio_target'},
    ]

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_diag_moment_source_summary.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_diag_moment_source_detail.csv'
    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with detail_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)
    print(summary_path)
    print(detail_path)


if __name__ == '__main__':
    main()
