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
    up_mass = p.m0 * p.m0 * (up_om * up_om - 1.0)
    um_mass = p.m0 * p.m0 * (um_om * um_om - 1.0)
    up_der = (1.0 - 6.0 * p.xi) * (lap_omega_single(rr, zz, D / 2.0, p) / up_om)
    um_der = (1.0 - 6.0 * p.xi) * (lap_omega_single(rr, zz, -D / 2.0, p) / um_om)

    delta_mass_p = up_mass - interp(_mass_bar(D / 2.0))
    delta_mass_m = um_mass - interp(_mass_bar(-D / 2.0))
    delta_der_p = up_der - interp(_der_bar(D / 2.0))
    delta_der_m = um_der - interp(_der_bar(-D / 2.0))

    q_diag = np.zeros_like(C)
    q_diag[plus] += C[plus] * (delta_mass_p[plus] + delta_der_p[plus])
    q_diag[minus] += C[minus] * (delta_mass_m[minus] + delta_der_m[minus])
    sign = 1.0 if float(np.sum(q_diag[anchor])) >= 0.0 else -1.0

    local_mass = np.zeros_like(C)
    local_mass[plus] = delta_mass_p[plus]
    local_mass[minus] = delta_mass_m[minus]

    oriented_carrier = sign * C
    carrier_pos = np.clip(oriented_carrier, 0.0, None)
    carrier_neg = np.clip(-oriented_carrier, 0.0, None)

    q_mass = oriented_carrier * local_mass
    q_mass_pos = np.clip(q_mass, 0.0, None)
    factored = local_mass * carrier_pos

    local_mass_anchor = local_mass[anchor]
    carrier_pos_anchor = carrier_pos[anchor]
    carrier_neg_anchor = carrier_neg[anchor]
    q_mass_pos_anchor = q_mass_pos[anchor]

    P_mass = float(np.sum(q_mass_pos_anchor))
    W_pos = float(np.sum(carrier_pos_anchor))
    W_neg = float(np.sum(carrier_neg_anchor))
    I_carrier = float(np.sum(oriented_carrier[anchor]))
    k_min = float(np.min(local_mass_anchor))
    k_max = float(np.max(local_mass_anchor))
    k_avg = P_mass / max(W_pos, 1e-300)
    factor_resid = float(np.max(np.abs(q_mass_pos_anchor - factored[anchor])))
    lower_product = k_min * W_pos
    upper_product = k_max * W_pos

    P_mass_summary = next(csv.DictReader((OUT_DIR / 'chi_open_system_exact_schur_singleton_diag_mass_bridge_source_summary.csv').open()))
    req_Pmass_global = float(P_mass_summary['required_Pmass_global_given_Nder'])
    req_Pmass_local = float(P_mass_summary['required_Pmass_local_given_Nder'])

    req_W_global = req_Pmass_global / max(k_avg, 1e-300)
    req_W_local = req_Pmass_local / max(k_avg, 1e-300)
    req_k_global = req_Pmass_global / max(W_pos, 1e-300)
    req_k_local = req_Pmass_local / max(W_pos, 1e-300)

    P_mass_plus = float(np.sum(q_mass_pos[plus]))
    P_mass_minus = float(np.sum(q_mass_pos[minus]))
    W_pos_plus = float(np.sum(carrier_pos[plus]))
    W_pos_minus = float(np.sum(carrier_pos[minus]))
    k_avg_plus = P_mass_plus / max(W_pos_plus, 1e-300)
    k_avg_minus = P_mass_minus / max(W_pos_minus, 1e-300)

    summary = {
        'D': D,
        'rho_anchor': rho_anchor,
        'zeta_anchor': zeta_anchor,
        'positive_kernel_everywhere_on_anchor': int(np.all(local_mass_anchor > 0.0)),
        'factorization_residual_abs': factor_resid,
        'P_mass_11': P_mass,
        'W_pos_11': W_pos,
        'W_neg_11': W_neg,
        'I_carrier_11': I_carrier,
        'carrier_neg_over_pos_11': W_neg / max(W_pos, 1e-300),
        'mass_kernel_min_anchor': k_min,
        'mass_kernel_max_anchor': k_max,
        'mass_kernel_avg_weighted': k_avg,
        'lower_product_from_kernel_min': lower_product,
        'upper_product_from_kernel_max': upper_product,
        'required_Pmass_global': req_Pmass_global,
        'required_Pmass_local': req_Pmass_local,
        'lower_product_over_required_global': lower_product / max(req_Pmass_global, 1e-300),
        'lower_product_over_required_local': lower_product / max(req_Pmass_local, 1e-300),
        'required_Wpos_global_given_kavg': req_W_global,
        'required_Wpos_local_given_kavg': req_W_local,
        'Wpos_over_required_global': W_pos / max(req_W_global, 1e-300),
        'Wpos_over_required_local': W_pos / max(req_W_local, 1e-300),
        'required_kavg_global_given_Wpos': req_k_global,
        'required_kavg_local_given_Wpos': req_k_local,
        'kavg_over_required_global': k_avg / max(req_k_global, 1e-300),
        'kavg_over_required_local': k_avg / max(req_k_local, 1e-300),
        'P_mass_plus_11': P_mass_plus,
        'P_mass_minus_11': P_mass_minus,
        'W_pos_plus_11': W_pos_plus,
        'W_pos_minus_11': W_pos_minus,
        'kavg_plus_11': k_avg_plus,
        'kavg_minus_11': k_avg_minus,
        'plus_minus_mass_rel_gap': abs(P_mass_plus - P_mass_minus) / max(P_mass, 1e-300),
        'plus_minus_carrier_rel_gap': abs(W_pos_plus - W_pos_minus) / max(W_pos, 1e-300),
        'plus_minus_kavg_rel_gap': abs(k_avg_plus - k_avg_minus) / max(k_avg, 1e-300),
        'preferred_parent_side_route': 'positive_oriented_carrier_moment',
    }

    detail_rows = [
        {'term': 'P_mass_11', 'value': P_mass, 'role': 'target_positive_mass_carrier'},
        {'term': 'W_pos_11', 'value': W_pos, 'role': 'positive_oriented_carrier_moment'},
        {'term': 'mass_kernel_avg_weighted', 'value': k_avg, 'role': 'weighted_mass_kernel_average'},
        {'term': 'mass_kernel_min_anchor', 'value': k_min, 'role': 'coarse_kernel_floor'},
        {'term': 'lower_product_from_kernel_min', 'value': lower_product, 'role': 'coarse_lower_product'},
        {'term': 'required_Pmass_global', 'value': req_Pmass_global, 'role': 'global_mass_target'},
        {'term': 'required_Pmass_local', 'value': req_Pmass_local, 'role': 'local_mass_target'},
        {'term': 'required_Wpos_global_given_kavg', 'value': req_W_global, 'role': 'global_carrier_target'},
        {'term': 'required_Wpos_local_given_kavg', 'value': req_W_local, 'role': 'local_carrier_target'},
        {'term': 'required_kavg_global_given_Wpos', 'value': req_k_global, 'role': 'global_weighted_kernel_target'},
        {'term': 'required_kavg_local_given_Wpos', 'value': req_k_local, 'role': 'local_weighted_kernel_target'},
        {'term': 'P_mass_plus_11', 'value': P_mass_plus, 'role': 'plus_lobe_mass'},
        {'term': 'P_mass_minus_11', 'value': P_mass_minus, 'role': 'minus_lobe_mass'},
        {'term': 'W_pos_plus_11', 'value': W_pos_plus, 'role': 'plus_lobe_carrier'},
        {'term': 'W_pos_minus_11', 'value': W_pos_minus, 'role': 'minus_lobe_carrier'},
        {'term': 'kavg_plus_11', 'value': k_avg_plus, 'role': 'plus_lobe_weighted_kernel'},
        {'term': 'kavg_minus_11', 'value': k_avg_minus, 'role': 'minus_lobe_weighted_kernel'},
    ]

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_pmass_kernel_source_summary.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_pmass_kernel_source_detail.csv'
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
