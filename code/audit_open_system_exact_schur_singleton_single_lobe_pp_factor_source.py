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

    phi_R = (psi1 + psi2) / np.sqrt(2.0)
    phi_L = (psi1 - psi2) / np.sqrt(2.0)
    phi_R_p = np.clip(phi_R, 0.0, None)
    phi_L_p = np.clip(phi_L, 0.0, None)

    density_pp = 2.0 * w[plus] * phi_R_p[plus] * phi_L_p[plus]
    W_pp = float(np.sum(density_pp))

    S_R_pos = float(np.sum(2.0 * w[plus] * phi_R_p[plus]))
    S_L_pos = float(np.sum(2.0 * w[plus] * phi_L_p[plus]))

    avg_phiL_given_Rpos = W_pp / max(S_R_pos, 1e-300)
    avg_phiR_given_Lpos = W_pp / max(S_L_pos, 1e-300)

    resid_R = abs(W_pp - S_R_pos * avg_phiL_given_Rpos)
    resid_L = abs(W_pp - S_L_pos * avg_phiR_given_Lpos)

    prior = next(csv.DictReader((OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_overlap_source_summary.csv').open()))
    req_half_global = float(prior['required_half_global'])
    req_half_local = float(prior['required_half_local'])

    req_S_L_global = req_half_global / max(avg_phiR_given_Lpos, 1e-300)
    req_S_L_local = req_half_local / max(avg_phiR_given_Lpos, 1e-300)
    req_S_R_global = req_half_global / max(avg_phiL_given_Rpos, 1e-300)
    req_S_R_local = req_half_local / max(avg_phiL_given_Rpos, 1e-300)

    req_avg_phiR_global = req_half_global / max(S_L_pos, 1e-300)
    req_avg_phiR_local = req_half_local / max(S_L_pos, 1e-300)
    req_avg_phiL_global = req_half_global / max(S_R_pos, 1e-300)
    req_avg_phiL_local = req_half_local / max(S_R_pos, 1e-300)

    mask_Lpos = plus & (phi_L > 0.0)
    phiR_positive_on_phiL_positive_support = int(np.all(phi_R[mask_Lpos] > 0.0)) if np.any(mask_Lpos) else 0
    min_phiR_on_phiL_positive_support = float(np.min(phi_R[mask_Lpos])) if np.any(mask_Lpos) else 0.0

    summary = {
        'D': D,
        'rho_anchor': rho_anchor,
        'zeta_anchor': zeta_anchor,
        'single_lobe_positive_positive_overlap': W_pp,
        'single_lobe_same_center_positive_mass': S_R_pos,
        'single_lobe_mirror_positive_mass': S_L_pos,
        'weighted_mirror_positive_avg_given_same_center': avg_phiL_given_Rpos,
        'weighted_same_center_avg_given_mirror_positive': avg_phiR_given_Lpos,
        'same_center_route_factorization_residual_abs': resid_R,
        'mirror_route_factorization_residual_abs': resid_L,
        'required_half_global': req_half_global,
        'required_half_local': req_half_local,
        'required_mirror_positive_mass_global': req_S_L_global,
        'required_mirror_positive_mass_local': req_S_L_local,
        'required_same_center_positive_mass_global': req_S_R_global,
        'required_same_center_positive_mass_local': req_S_R_local,
        'required_weighted_same_center_avg_global': req_avg_phiR_global,
        'required_weighted_same_center_avg_local': req_avg_phiR_local,
        'required_weighted_mirror_avg_global': req_avg_phiL_global,
        'required_weighted_mirror_avg_local': req_avg_phiL_local,
        'mirror_mass_over_required_global': S_L_pos / max(req_S_L_global, 1e-300),
        'mirror_mass_over_required_local': S_L_pos / max(req_S_L_local, 1e-300),
        'same_center_mass_over_required_global': S_R_pos / max(req_S_R_global, 1e-300),
        'same_center_mass_over_required_local': S_R_pos / max(req_S_R_local, 1e-300),
        'weighted_same_center_over_required_global': avg_phiR_given_Lpos / max(req_avg_phiR_global, 1e-300),
        'weighted_same_center_over_required_local': avg_phiR_given_Lpos / max(req_avg_phiR_local, 1e-300),
        'weighted_mirror_over_required_global': avg_phiL_given_Rpos / max(req_avg_phiL_global, 1e-300),
        'weighted_mirror_over_required_local': avg_phiL_given_Rpos / max(req_avg_phiL_local, 1e-300),
        'phiR_positive_on_phiL_positive_support': phiR_positive_on_phiL_positive_support,
        'min_phiR_on_phiL_positive_support': min_phiR_on_phiL_positive_support,
        'preferred_parent_side_route': 'single_lobe_mirror_positive_mass',
    }

    detail_rows = [
        {'term': 'single_lobe_positive_positive_overlap', 'value': W_pp, 'role': 'target_positive_positive_overlap'},
        {'term': 'single_lobe_same_center_positive_mass', 'value': S_R_pos, 'role': 'same_center_positive_lobe_mass'},
        {'term': 'single_lobe_mirror_positive_mass', 'value': S_L_pos, 'role': 'mirror_positive_lobe_mass'},
        {'term': 'weighted_mirror_positive_avg_given_same_center', 'value': avg_phiL_given_Rpos, 'role': 'mirror_multiplier_given_same_center'},
        {'term': 'weighted_same_center_avg_given_mirror_positive', 'value': avg_phiR_given_Lpos, 'role': 'same_center_multiplier_given_mirror'},
        {'term': 'required_half_global', 'value': req_half_global, 'role': 'global_single_lobe_pp_threshold'},
        {'term': 'required_half_local', 'value': req_half_local, 'role': 'local_single_lobe_pp_threshold'},
        {'term': 'required_mirror_positive_mass_global', 'value': req_S_L_global, 'role': 'global_mirror_mass_floor'},
        {'term': 'required_mirror_positive_mass_local', 'value': req_S_L_local, 'role': 'local_mirror_mass_floor'},
    ]

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_pp_factor_source_summary.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_single_lobe_pp_factor_source_detail.csv'
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
