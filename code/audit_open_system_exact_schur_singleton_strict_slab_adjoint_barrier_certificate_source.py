#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh, spsolve

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from extract_chi_localized_2d import Level, PhysicalParams, build_generalized_operator, normalize_modes

OUT_DIR = ROOT / 'output' / 'chi_open_system'
B_SIG_FIGS = 4
P_SIG_FIGS = 4
BARRIER_THETA = 0.999999


def floor_sig(x: float, sig_figs: int) -> float:
    if x <= 0.0:
        return 0.0
    scale = 10.0 ** (math.floor(math.log10(abs(x))) - sig_figs + 1)
    return math.floor(x / scale) * scale


def main() -> None:
    strict_summary = next(
        csv.DictReader(
            (OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_strict_slab_source_summary.csv').open()
        )
    )
    dirichlet_summary = next(
        csv.DictReader(
            (
                OUT_DIR
                / 'chi_open_system_exact_schur_singleton_strict_slab_dirichlet_split_source_summary.csv'
            ).open()
        )
    )

    D = float(strict_summary['D'])
    rho_anchor = float(strict_summary['rho_anchor'])
    zeta_anchor = float(strict_summary['zeta_anchor'])
    h_star = float(strict_summary['strict_cap'])
    required_global = float(strict_summary['required_global'])
    required_local = float(strict_summary['required_local'])
    source = float(dirichlet_summary['strict_slab_source_potential_mass'])
    target_global_after_source = required_global - source
    target_local_after_source = required_local - source

    p = PhysicalParams()
    level = Level('fine', dr=0.06, dz=0.03)
    z_max = D / 2.0 + 6.0

    rho, z, rr, zz, uu, k_mat, m_mat = build_generalized_operator(
        D=D,
        p=p,
        rho_max=3.0,
        z_max=z_max,
        dr=level.dr,
        dz=level.dz,
    )
    k_mat = k_mat.tocsr()
    m_mat = m_mat.tocsr()

    evals, evecs = eigsh(k_mat, k=2, M=m_mat, sigma=2.5, which='LM', tol=1e-8, maxiter=30000)
    idx = np.argsort(np.real(evals))
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])

    psi = np.stack(
        [evecs[:, 0].reshape(len(rho), len(z)), evecs[:, 1].reshape(len(rho), len(z))],
        axis=2,
    )
    psi = normalize_modes(psi, rho, level.dr, level.dz)
    phi_L = (psi[:, :, 0] - psi[:, :, 1]) / np.sqrt(2.0)
    phi_R = (psi[:, :, 0] + psi[:, :, 1]) / np.sqrt(2.0)

    w = 2.0 * np.pi * rr * level.dr * level.dz
    plus = (rr <= rho_anchor) & (np.abs(zz - D / 2.0) <= zeta_anchor)
    strict = plus & ((zz - D / 2.0) <= -h_star)

    ebar = 0.5 * (evals[0] + evals[1])
    delta_e = 0.5 * (evals[1] - evals[0])
    rhs0 = (ebar - uu) * phi_L - delta_e * phi_R
    klap = (k_mat - diags((rr * uu).ravel(), 0, format='csr')).tocsr()

    flat = np.flatnonzero(strict.ravel())
    local_of_global = -np.ones(strict.size, dtype=int)
    local_of_global[flat] = np.arange(len(flat))

    ksub = klap[flat][:, flat].tocsr()
    msub = m_mat[flat][:, flat].tocsr()
    phi_flat = phi_L.ravel()[flat]
    rhs_flat = rhs0.ravel()[flat]
    boundary_defect = ksub @ phi_flat - msub @ rhs_flat
    weight_flat = (2.0 * w).ravel()[flat]

    poisson_weight = spsolve(ksub.T, weight_flat)
    barrier_weight = BARRIER_THETA * weight_flat
    barrier_poisson = spsolve(ksub.T, barrier_weight)
    barrier_residual = weight_flat - ksub.T @ barrier_poisson
    exact_relation_residual = barrier_poisson - BARRIER_THETA * poisson_weight

    up_in = np.zeros_like(strict, dtype=bool)
    up_in[:, :-1] = strict[:, 1:]
    down_in = np.zeros_like(strict, dtype=bool)
    down_in[:, 1:] = strict[:, :-1]
    right_in = np.zeros_like(strict, dtype=bool)
    right_in[:-1, :] = strict[1:, :]

    ii, _ = np.indices(strict.shape)
    boundary = strict & (~up_in | ~down_in | ~right_in | (ii == 0))
    on_axis = strict & (ii == 0)
    on_top = strict & ~up_in
    on_bottom = strict & ~down_in
    on_outer = strict & ~right_in

    categories = {
        'outer_rho': boundary & on_outer & ~on_top & ~on_bottom,
        'bottom': boundary & on_bottom & ~on_axis & ~on_outer,
        'corner_rb': boundary & on_outer & on_bottom,
        'corner_lb': boundary & on_axis & on_bottom,
    }
    labels = {
        'outer_rho': 'out',
        'bottom': 'bot',
        'corner_rb': 'rb',
        'corner_lb': 'lb',
    }
    edge_order = ['outer_rho', 'bottom', 'corner_rb', 'corner_lb']

    point_rows = []
    edge_totals: dict[str, dict[str, float]] = {}
    all_valid = True

    for edge in edge_order:
        global_idx = np.flatnonzero(categories[edge].ravel())
        global_idx = sorted(
            global_idx,
            key=lambda idx_flat: (
                z[np.unravel_index(idx_flat, strict.shape)[1]],
                rho[np.unravel_index(idx_flat, strict.shape)[0]],
            ),
        )
        edge_exact = 0.0
        edge_cert = 0.0
        edge_barrier = 0.0
        edge_b_floor = 0.0
        for point_id, idx_flat in enumerate(global_idx, start=1):
            i_rho, i_z = np.unravel_index(idx_flat, strict.shape)
            local_idx = local_of_global[idx_flat]
            b_exact = float(boundary_defect[local_idx])
            p_exact = float(poisson_weight[local_idx])
            p_barrier = float(barrier_poisson[local_idx])
            b_floor = floor_sig(b_exact, B_SIG_FIGS)
            p_floor = floor_sig(p_exact, P_SIG_FIGS)
            exact_product = b_exact * p_exact
            barrier_product = b_floor * p_barrier
            certified_product = b_floor * p_floor
            valid = (
                b_exact >= b_floor
                and p_barrier >= p_floor
                and p_exact >= p_barrier
                and b_floor >= 0.0
                and p_floor >= 0.0
            )
            all_valid = all_valid and valid
            edge_exact += exact_product
            edge_barrier += barrier_product
            edge_cert += certified_product
            edge_b_floor += b_floor
            point_rows.append(
                {
                    'edge': edge,
                    'edge_label': labels[edge],
                    'point_id': point_id,
                    'rho': float(rho[i_rho]),
                    'z_rel': float(z[i_z] - D / 2.0),
                    'exact_boundary_defect': b_exact,
                    'certified_boundary_defect_floor': b_floor,
                    'exact_poisson_weight': p_exact,
                    'barrier_poisson_weight': p_barrier,
                    'certified_poisson_weight_floor': p_floor,
                    'exact_minus_barrier_weight': p_exact - p_barrier,
                    'barrier_minus_certified_weight': p_barrier - p_floor,
                    'exact_product': exact_product,
                    'barrier_product_with_b_floor': barrier_product,
                    'certified_product_floor': certified_product,
                    'product_slack_exact_minus_certified': exact_product - certified_product,
                    'point_barrier_certificate_valid': valid,
                }
            )
        edge_totals[edge] = {
            'exact_product': edge_exact,
            'barrier_product_with_b_floor': edge_barrier,
            'certified_product_floor': edge_cert,
            'certified_boundary_defect_sum': edge_b_floor,
        }

    edge_rows = []
    cumulative_cert = 0.0
    cumulative_barrier = 0.0
    for edge in edge_order:
        row = edge_totals[edge]
        cumulative_cert += row['certified_product_floor']
        cumulative_barrier += row['barrier_product_with_b_floor']
        edge_rows.append(
            {
                'edge': edge,
                'edge_label': labels[edge],
                'exact_product': row['exact_product'],
                'barrier_product_with_b_floor': row['barrier_product_with_b_floor'],
                'certified_product_floor': row['certified_product_floor'],
                'exact_minus_certified_product': row['exact_product'] - row['certified_product_floor'],
                'barrier_minus_certified_product': row['barrier_product_with_b_floor']
                - row['certified_product_floor'],
                'cumulative_barrier_product': cumulative_barrier,
                'cumulative_certified_product': cumulative_cert,
                'cumulative_certified_over_global_after_source': cumulative_cert
                / max(target_global_after_source, 1e-300),
                'cumulative_certified_over_local_after_source': cumulative_cert
                / max(target_local_after_source, 1e-300),
            }
        )

    exact_edge_sum = sum(row['exact_product'] for row in edge_totals.values())
    barrier_edge_sum = sum(row['barrier_product_with_b_floor'] for row in edge_totals.values())
    certified_edge_sum = sum(row['certified_product_floor'] for row in edge_totals.values())
    certified_outer_bottom = (
        edge_totals['outer_rho']['certified_product_floor']
        + edge_totals['bottom']['certified_product_floor']
    )
    certified_outer_bottom_rb = certified_outer_bottom + edge_totals['corner_rb']['certified_product_floor']
    gap_after_outer_bottom_rb = target_global_after_source - certified_outer_bottom_rb
    certified_total_with_source = certified_edge_sum + source

    summary = {
        'D': D,
        'rho_anchor': rho_anchor,
        'zeta_anchor': zeta_anchor,
        'strict_cap': h_star,
        'boundary_defect_sig_figs': B_SIG_FIGS,
        'poisson_weight_sig_figs': P_SIG_FIGS,
        'barrier_theta': BARRIER_THETA,
        'required_global': required_global,
        'required_local': required_local,
        'source_potential_mass': source,
        'target_global_after_source': target_global_after_source,
        'target_local_after_source': target_local_after_source,
        'exact_edge_product_sum': exact_edge_sum,
        'barrier_product_with_b_floor_sum': barrier_edge_sum,
        'certified_barrier_edge_sum': certified_edge_sum,
        'certified_barrier_edge_sum_over_global_after_source': certified_edge_sum
        / max(target_global_after_source, 1e-300),
        'certified_barrier_edge_sum_over_local_after_source': certified_edge_sum
        / max(target_local_after_source, 1e-300),
        'certified_barrier_gap_over_global_after_source': certified_edge_sum - target_global_after_source,
        'exact_minus_certified_barrier_edge_sum': exact_edge_sum - certified_edge_sum,
        'barrier_minus_certified_edge_sum': barrier_edge_sum - certified_edge_sum,
        'certified_total_with_source': certified_total_with_source,
        'certified_total_over_required_global': certified_total_with_source / max(required_global, 1e-300),
        'certified_total_over_required_local': certified_total_with_source / max(required_local, 1e-300),
        'certified_total_gap_over_required_global': certified_total_with_source - required_global,
        'certified_outer_bottom_product': certified_outer_bottom,
        'certified_outer_bottom_over_local_after_source': certified_outer_bottom
        / max(target_local_after_source, 1e-300),
        'certified_outer_bottom_right_corner_product': certified_outer_bottom_rb,
        'certified_gap_after_outer_bottom_right_corner': gap_after_outer_bottom_rb,
        'certified_left_corner_product': edge_totals['corner_lb']['certified_product_floor'],
        'certified_left_corner_over_gap_after_outer_bottom_right_corner': edge_totals['corner_lb'][
            'certified_product_floor'
        ]
        / max(gap_after_outer_bottom_rb, 1e-300),
        'barrier_residual_min': float(np.min(barrier_residual)),
        'barrier_residual_max': float(np.max(barrier_residual)),
        'barrier_residual_expected_min': float((1.0 - BARRIER_THETA) * np.min(weight_flat)),
        'barrier_residual_expected_max': float((1.0 - BARRIER_THETA) * np.max(weight_flat)),
        'barrier_relation_resid_abs_max': float(np.max(np.abs(exact_relation_residual))),
        'barrier_is_subsolution': int(np.all(barrier_residual >= -1e-13)),
        'point_barrier_certificate_valid': all_valid,
        'certificate_route': 'theta_scaled_adjoint_poisson_barrier_with_4sig_b_and_4sig_p_floors',
    }

    summary_path = (
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_adjoint_barrier_certificate_source_summary.csv'
    )
    edge_path = (
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_adjoint_barrier_certificate_source_edges.csv'
    )
    detail_path = (
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_adjoint_barrier_certificate_source_detail.csv'
    )

    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with edge_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edge_rows[0].keys()))
        writer.writeheader()
        for row in edge_rows:
            writer.writerow(row)
    with detail_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0].keys()))
        writer.writeheader()
        for row in point_rows:
            writer.writerow(row)

    print(summary_path)
    print(edge_path)
    print(detail_path)


if __name__ == '__main__':
    main()
