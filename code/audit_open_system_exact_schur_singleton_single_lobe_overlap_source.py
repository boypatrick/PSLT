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
    contrast = f['contrast_density']
    w = 2.0 * np.pi * rr * level.dr * level.dz

    plus = (rr <= rho_anchor) & (np.abs(zz - D / 2.0) <= zeta_anchor)
    minus = (rr <= rho_anchor) & (np.abs(zz + D / 2.0) <= zeta_anchor)
    anchor = plus | minus
    sign = 1.0 if float(np.sum(contrast[anchor])) >= 0.0 else -1.0
    oriented = sign * contrast

    phi_R = (psi1 + psi2) / np.sqrt(2.0)
    phi_L = (psi1 - psi2) / np.sqrt(2.0)

    phi_R_p = np.clip(phi_R, 0.0, None)
    phi_R_m = np.clip(-phi_R, 0.0, None)
    phi_L_p = np.clip(phi_L, 0.0, None)
    phi_L_m = np.clip(-phi_L, 0.0, None)

    density_exact_plus = np.clip(oriented[plus], 0.0, None)
    density_pp_plus = 2.0 * w[plus] * phi_R_p[plus] * phi_L_p[plus]
    density_mm_plus = 2.0 * w[plus] * phi_R_m[plus] * phi_L_m[plus]
    density_exact_from_pm = density_pp_plus + density_mm_plus

    W_plus_exact = float(np.sum(density_exact_plus))
    W_pp = float(np.sum(density_pp_plus))
    W_mm = float(np.sum(density_mm_plus))
    factor_resid = float(np.max(np.abs(density_exact_plus - density_exact_from_pm))) if density_exact_plus.size else 0.0

    A_pp = float(np.sum(w[plus] * phi_R_p[plus] * phi_L_p[plus]))
    B_pm = float(np.sum(w[plus] * phi_R_p[plus] * phi_L_m[plus]))
    C_mp = float(np.sum(w[plus] * phi_R_m[plus] * phi_L_p[plus]))
    D_mm = float(np.sum(w[plus] * phi_R_m[plus] * phi_L_m[plus]))

    single_lobe = next(csv.DictReader((OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_carrier_source_summary.csv').open()))
    req_half_global = float(single_lobe['required_half_global'])
    req_half_local = float(single_lobe['required_half_local'])

    phiR_pos_L1 = float(np.sum(w[plus] * phi_R_p[plus]))
    phiR_neg_L1 = float(np.sum(w[plus] * phi_R_m[plus]))

    summary = {
        'D': D,
        'rho_anchor': rho_anchor,
        'zeta_anchor': zeta_anchor,
        'single_lobe_positive_overlap_exact': W_plus_exact,
        'single_lobe_positive_positive_overlap': W_pp,
        'single_lobe_negative_negative_completion': W_mm,
        'positive_overlap_split_residual_abs': factor_resid,
        'A_pp': A_pp,
        'B_pm': B_pm,
        'C_mp': C_mp,
        'D_mm': D_mm,
        'C_mp_is_zero': int(abs(C_mp) < 1e-18),
        'required_half_global': req_half_global,
        'required_half_local': req_half_local,
        'pp_over_required_half_global': W_pp / max(req_half_global, 1e-300),
        'pp_over_required_half_local': W_pp / max(req_half_local, 1e-300),
        'exact_over_required_half_global': W_plus_exact / max(req_half_global, 1e-300),
        'exact_over_required_half_local': W_plus_exact / max(req_half_local, 1e-300),
        'mm_over_exact': W_mm / max(W_plus_exact, 1e-300),
        'pp_over_exact': W_pp / max(W_plus_exact, 1e-300),
        'phiR_neg_over_pos_L1': phiR_neg_L1 / max(phiR_pos_L1, 1e-300),
        'phiR_neg_point_fraction': float(np.mean(phi_R[plus] < 0.0)),
        'phiL_pos_point_fraction': float(np.mean(phi_L[plus] > 0.0)),
        'preferred_parent_side_route': 'single_lobe_positive_positive_overlap',
    }

    detail_rows = [
        {'term': 'single_lobe_positive_overlap_exact', 'value': W_plus_exact, 'role': 'target_single_lobe_positive_carrier'},
        {'term': 'single_lobe_positive_positive_overlap', 'value': W_pp, 'role': 'positive_positive_overlap_lower_bound'},
        {'term': 'single_lobe_negative_negative_completion', 'value': W_mm, 'role': 'same_sign_completion'},
        {'term': 'required_half_global', 'value': req_half_global, 'role': 'global_single_lobe_threshold'},
        {'term': 'required_half_local', 'value': req_half_local, 'role': 'local_single_lobe_threshold'},
        {'term': 'A_pp', 'value': A_pp, 'role': 'plusplus_quadrant_overlap'},
        {'term': 'B_pm', 'value': B_pm, 'role': 'plusminus_quadrant_overlap'},
        {'term': 'C_mp', 'value': C_mp, 'role': 'minusplus_quadrant_overlap'},
        {'term': 'D_mm', 'value': D_mm, 'role': 'minusminus_quadrant_overlap'},
        {'term': 'phiR_neg_over_pos_L1', 'value': summary['phiR_neg_over_pos_L1'], 'role': 'same_center_negative_lobe_share'},
    ]

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_overlap_source_summary.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_overlap_source_detail.csv'
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
