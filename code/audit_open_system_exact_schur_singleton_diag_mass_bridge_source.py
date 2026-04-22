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

from audit_open_system_parity_contrast_self_mirror_source import solve_fields, omega_single, lap_omega_single, spherical_average_generic
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
    C = f['contrast_density']
    r_field = f['r_field']
    r_line = np.linspace(0.0, float(r_field.max()), 1800)
    interp = lambda arr: np.interp(r_field.ravel(), r_line, arr).reshape(r_field.shape)

    plus = (rr <= rho_anchor) & (np.abs(zz - D / 2.0) <= zeta_anchor)
    minus = (rr <= rho_anchor) & (np.abs(zz + D / 2.0) <= zeta_anchor)
    anchor = plus | minus

    def _mass_bar(center: float) -> np.ndarray:
        return spherical_average_generic(
            r_line,
            D,
            p,
            lambda rho, z, D, p: p.m0 * p.m0 * (omega_single(rho, z, center, p) ** 2 - 1.0),
            120,
        )

    def _der_bar(center: float) -> np.ndarray:
        return spherical_average_generic(
            r_line,
            D,
            p,
            lambda rho, z, D, p: (1.0 - 6.0 * p.xi) * (lap_omega_single(rho, z, center, p) / omega_single(rho, z, center, p)),
            120,
        )

    up_om = omega_single(rr, zz, D / 2.0, p)
    um_om = omega_single(rr, zz, -D / 2.0, p)
    up_lap = lap_omega_single(rr, zz, D / 2.0, p)
    um_lap = lap_omega_single(rr, zz, -D / 2.0, p)

    up_mass = p.m0 * p.m0 * (up_om * up_om - 1.0)
    um_mass = p.m0 * p.m0 * (um_om * um_om - 1.0)
    up_der = (1.0 - 6.0 * p.xi) * (up_lap / up_om)
    um_der = (1.0 - 6.0 * p.xi) * (um_lap / um_om)

    delta_mass_p = up_mass - interp(_mass_bar(D / 2.0))
    delta_mass_m = um_mass - interp(_mass_bar(-D / 2.0))
    delta_der_p = up_der - interp(_der_bar(D / 2.0))
    delta_der_m = um_der - interp(_der_bar(-D / 2.0))

    q_mass = np.zeros_like(C)
    q_der = np.zeros_like(C)
    q_diag = np.zeros_like(C)
    q_mass[plus] += C[plus] * delta_mass_p[plus]
    q_mass[minus] += C[minus] * delta_mass_m[minus]
    q_der[plus] += C[plus] * delta_der_p[plus]
    q_der[minus] += C[minus] * delta_der_m[minus]
    q_diag = q_mass + q_der

    sign = 1.0 if float(np.sum(q_diag[anchor])) >= 0.0 else -1.0
    q_mass_o = sign * q_mass[anchor]
    q_der_o = sign * q_der[anchor]
    q_diag_o = sign * q_diag[anchor]

    P_mass = float(np.sum(np.clip(q_mass_o, 0.0, None)))
    N_mass = float(np.sum(np.clip(-q_mass_o, 0.0, None)))
    P_der = float(np.sum(np.clip(q_der_o, 0.0, None)))
    N_der = float(np.sum(np.clip(-q_der_o, 0.0, None)))
    P_diag = float(np.sum(np.clip(q_diag_o, 0.0, None)))
    N_diag = float(np.sum(np.clip(-q_diag_o, 0.0, None)))
    I_diag = float(np.sum(q_diag_o))

    pointwise_rhs = np.clip(q_mass_o, 0.0, None) - np.clip(-q_der_o, 0.0, None)
    pointwise_violation = float(np.max(pointwise_rhs - np.clip(q_diag_o, 0.0, None)))
    induced_Pdiag_lower = P_mass - N_der

    diag_summary = next(csv.DictReader((OUT_DIR / 'chi_open_system_exact_schur_singleton_diag_moment_source_summary.csv').open()))
    req_Pdiag_global = float(diag_summary['required_P_diag_global_given_nu11'])
    req_Pdiag_local = float(diag_summary['required_P_diag_local_given_nu11'])

    req_Pmass_global = req_Pdiag_global + N_der
    req_Pmass_local = req_Pdiag_local + N_der
    allowed_Nder_global = P_mass - req_Pdiag_global
    allowed_Nder_local = P_mass - req_Pdiag_local

    summary = {
        'D': D,
        'rho_anchor': rho_anchor,
        'zeta_anchor': zeta_anchor,
        'P_mass_11': P_mass,
        'N_mass_11': N_mass,
        'P_der_11': P_der,
        'N_der_11': N_der,
        'P_diag_11': P_diag,
        'N_diag_11': N_diag,
        'I_diag_11': I_diag,
        'pointwise_mass_minus_nder_violation': pointwise_violation,
        'induced_Pdiag_lower_from_mass_minus_nder': induced_Pdiag_lower,
        'required_Pdiag_global': req_Pdiag_global,
        'required_Pdiag_local': req_Pdiag_local,
        'induced_over_required_global': induced_Pdiag_lower / max(req_Pdiag_global, 1e-300),
        'induced_over_required_local': induced_Pdiag_lower / max(req_Pdiag_local, 1e-300),
        'required_Pmass_global_given_Nder': req_Pmass_global,
        'required_Pmass_local_given_Nder': req_Pmass_local,
        'Pmass_over_required_global': P_mass / max(req_Pmass_global, 1e-300),
        'Pmass_over_required_local': P_mass / max(req_Pmass_local, 1e-300),
        'allowed_Nder_global_given_Pmass': allowed_Nder_global,
        'allowed_Nder_local_given_Pmass': allowed_Nder_local,
        'Nder_over_allowed_global': N_der / max(allowed_Nder_global, 1e-300),
        'Nder_over_allowed_local': N_der / max(allowed_Nder_local, 1e-300),
        'nder_over_pmass': N_der / max(P_mass, 1e-300),
        'preferred_parent_side_route': 'P_mass_lower_bound',
    }

    detail_rows = [
        {'term': 'P_mass_11', 'value': P_mass, 'role': 'positive_mass_carrier'},
        {'term': 'N_der_11', 'value': N_der, 'role': 'derivative_adverse_leakage'},
        {'term': 'induced_Pdiag_lower_from_mass_minus_nder', 'value': induced_Pdiag_lower, 'role': 'diag_positive_moment_lower_bound'},
        {'term': 'required_Pdiag_global', 'value': req_Pdiag_global, 'role': 'global_positive_moment_target'},
        {'term': 'required_Pdiag_local', 'value': req_Pdiag_local, 'role': 'local_positive_moment_target'},
        {'term': 'required_Pmass_global_given_Nder', 'value': req_Pmass_global, 'role': 'global_mass_target'},
        {'term': 'required_Pmass_local_given_Nder', 'value': req_Pmass_local, 'role': 'local_mass_target'},
        {'term': 'allowed_Nder_global_given_Pmass', 'value': allowed_Nder_global, 'role': 'global_derivative_ceiling'},
        {'term': 'allowed_Nder_local_given_Pmass', 'value': allowed_Nder_local, 'role': 'local_derivative_ceiling'},
    ]

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_diag_mass_bridge_source_summary.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_diag_mass_bridge_source_detail.csv'
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
