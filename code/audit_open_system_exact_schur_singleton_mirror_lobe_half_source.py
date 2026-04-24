#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from audit_open_system_parity_contrast_self_mirror_source import solve_fields
from extract_chi_localized_2d import Level, PhysicalParams

OUT_DIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    D = 11.0
    rho_anchor = 0.70
    zeta_anchor = 1.21

    p = PhysicalParams()
    level = Level('fine', dr=0.06, dz=0.03)
    f = solve_fields(D=D, p=p, level=level, rho_max=3.0, z_margin=6.0, sigma=2.5, tol=1e-8, maxiter=30000)

    rr = f['rr']
    zz = f['zz']
    psi1 = f['psi1']
    psi2 = f['psi2']
    w = 2.0 * np.pi * rr * level.dr * level.dz

    plus = (rr <= rho_anchor) & (np.abs(zz - D / 2.0) <= zeta_anchor)
    lower = plus & (zz < D / 2.0)
    upper = plus & (zz >= D / 2.0)

    phi_L = (psi1 - psi2) / np.sqrt(2.0)
    phiL_pos = np.clip(phi_L, 0.0, None)
    phiL_neg = np.clip(-phi_L, 0.0, None)

    S_L_pos = float(np.sum(2.0 * w[plus] * phiL_pos[plus]))
    S_L_pos_lower = float(np.sum(2.0 * w[lower] * phiL_pos[lower]))
    S_L_pos_upper = float(np.sum(2.0 * w[upper] * phiL_pos[upper]))

    I_L_lower = float(np.sum(2.0 * w[lower] * phi_L[lower]))
    N_L_lower = float(np.sum(2.0 * w[lower] * phiL_neg[lower]))
    I_L_plus = float(np.sum(2.0 * w[plus] * phi_L[plus]))
    I_L_upper = float(np.sum(2.0 * w[upper] * phi_L[upper]))

    bridge_resid = abs(S_L_pos - (I_L_lower + N_L_lower))

    prior = next(csv.DictReader((OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_pp_factor_source_summary.csv').open()))
    req_global = float(prior['required_mirror_positive_mass_global'])
    req_local = float(prior['required_mirror_positive_mass_local'])

    z_rel = zz[plus] - D / 2.0
    row_data = []
    for zv in sorted(np.unique(np.round(z_rel, 6))):
        mask = plus & (np.round(zz - D / 2.0, 6) == zv)
        row_data.append({
            'z_rel': float(zv),
            'positive_mass': float(np.sum(2.0 * w[mask] * phiL_pos[mask])),
            'negative_mass': float(np.sum(2.0 * w[mask] * phiL_neg[mask])),
            'signed_mass': float(np.sum(2.0 * w[mask] * phi_L[mask])),
            'positive_point_fraction': float(np.mean(phi_L[mask] > 0.0)),
        })

    summary = {
        'D': D,
        'rho_anchor': rho_anchor,
        'zeta_anchor': zeta_anchor,
        'mirror_positive_lobe_mass': S_L_pos,
        'mirror_positive_lower_half_mass': S_L_pos_lower,
        'mirror_positive_upper_half_mass': S_L_pos_upper,
        'mirror_signed_lower_half_mass': I_L_lower,
        'mirror_negative_leakage_lower_half': N_L_lower,
        'mirror_signed_plus_lobe_mass': I_L_plus,
        'mirror_signed_upper_half_mass': I_L_upper,
        'lower_half_bridge_residual_abs': bridge_resid,
        'required_global': req_global,
        'required_local': req_local,
        'signed_lower_over_required_global': I_L_lower / max(req_global, 1e-300),
        'signed_lower_over_required_local': I_L_lower / max(req_local, 1e-300),
        'positive_mass_over_required_global': S_L_pos / max(req_global, 1e-300),
        'positive_mass_over_required_local': S_L_pos / max(req_local, 1e-300),
        'negative_leakage_over_positive_mass': N_L_lower / max(S_L_pos, 1e-300),
        'negative_leakage_over_signed_lower': N_L_lower / max(I_L_lower, 1e-300),
        'upper_half_is_nonpositive': int(np.all(phi_L[upper] <= 0.0)),
        'lower_positive_point_fraction': float(np.mean(phi_L[lower] > 0.0)),
        'min_phiL_lower': float(np.min(phi_L[lower])),
        'max_phiL_upper': float(np.max(phi_L[upper])),
        'preferred_parent_side_route': 'mirror_signed_lower_half_mass',
    }

    detail_rows = [
        {'term': 'mirror_positive_lobe_mass', 'value': S_L_pos, 'role': 'target_positive_mass'},
        {'term': 'mirror_signed_lower_half_mass', 'value': I_L_lower, 'role': 'signed_lower_half_candidate'},
        {'term': 'mirror_negative_leakage_lower_half', 'value': N_L_lower, 'role': 'lower_half_negative_leakage'},
        {'term': 'required_global', 'value': req_global, 'role': 'global_mirror_positive_threshold'},
        {'term': 'required_local', 'value': req_local, 'role': 'local_mirror_positive_threshold'},
        {'term': 'mirror_signed_upper_half_mass', 'value': I_L_upper, 'role': 'upper_half_signed_mass'},
    ]

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_lobe_half_source_summary.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_lobe_half_source_detail.csv'
    rows_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_lobe_half_source_rows.csv'
    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with detail_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)
    with rows_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_data[0].keys()))
        writer.writeheader()
        for row in row_data:
            writer.writerow(row)
    print(summary_path)
    print(detail_path)
    print(rows_path)


if __name__ == '__main__':
    main()
