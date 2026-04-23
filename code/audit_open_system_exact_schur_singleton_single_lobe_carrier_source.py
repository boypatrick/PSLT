#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    row = next(csv.DictReader((OUT_DIR / 'chi_open_system_exact_schur_singleton_pmass_kernel_source_summary.csv').open()))

    D = float(row['D'])
    rho_anchor = float(row['rho_anchor'])
    zeta_anchor = float(row['zeta_anchor'])
    W_total = float(row['W_pos_11'])
    W_plus = float(row['W_pos_plus_11'])
    W_minus = float(row['W_pos_minus_11'])
    half_total = 0.5 * W_total
    req_half_global = 0.5 * float(row['required_Wpos_global_given_kavg'])
    req_half_local = 0.5 * float(row['required_Wpos_local_given_kavg'])

    lobe_sum_resid = abs((W_plus + W_minus) - W_total)
    plus_half_resid = abs(W_plus - half_total)
    minus_half_resid = abs(W_minus - half_total)
    plus_minus_abs_gap = abs(W_plus - W_minus)

    summary = {
        'D': D,
        'rho_anchor': rho_anchor,
        'zeta_anchor': zeta_anchor,
        'W_pos_11': W_total,
        'W_pos_plus_11': W_plus,
        'W_pos_minus_11': W_minus,
        'half_total_W_pos_11': half_total,
        'required_half_global': req_half_global,
        'required_half_local': req_half_local,
        'lobe_sum_reconstruction_residual_abs': lobe_sum_resid,
        'plus_half_residual_abs': plus_half_resid,
        'minus_half_residual_abs': minus_half_resid,
        'plus_minus_abs_gap': plus_minus_abs_gap,
        'plus_minus_rel_total_gap': plus_minus_abs_gap / max(W_total, 1e-300),
        'plus_over_half_total': W_plus / max(half_total, 1e-300),
        'minus_over_half_total': W_minus / max(half_total, 1e-300),
        'plus_over_required_half_global': W_plus / max(req_half_global, 1e-300),
        'plus_over_required_half_local': W_plus / max(req_half_local, 1e-300),
        'minus_over_required_half_global': W_minus / max(req_half_global, 1e-300),
        'minus_over_required_half_local': W_minus / max(req_half_local, 1e-300),
        'single_lobe_target_route': 'positive_oriented_carrier_single_lobe',
    }

    detail_rows = [
        {'term': 'W_pos_11', 'value': W_total, 'role': 'total_positive_oriented_carrier'},
        {'term': 'W_pos_plus_11', 'value': W_plus, 'role': 'plus_lobe_positive_carrier'},
        {'term': 'W_pos_minus_11', 'value': W_minus, 'role': 'minus_lobe_positive_carrier'},
        {'term': 'half_total_W_pos_11', 'value': half_total, 'role': 'exact_symmetric_single_lobe_value'},
        {'term': 'required_half_global', 'value': req_half_global, 'role': 'global_single_lobe_threshold'},
        {'term': 'required_half_local', 'value': req_half_local, 'role': 'local_single_lobe_threshold'},
        {'term': 'plus_half_residual_abs', 'value': plus_half_resid, 'role': 'plus_vs_half_total_residual'},
        {'term': 'minus_half_residual_abs', 'value': minus_half_resid, 'role': 'minus_vs_half_total_residual'},
        {'term': 'plus_minus_abs_gap', 'value': plus_minus_abs_gap, 'role': 'lobe_gap'},
    ]

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_carrier_source_summary.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_carrier_source_detail.csv'
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
