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
PHI_SIG_FIGS = 4
P_SIG_FIGS = 4
PHI_BARRIER_THETA = 0.9999995


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

    def flat_index(i_rho: int, i_z: int) -> int:
        return np.ravel_multi_index((i_rho, i_z), strict.shape)

    def add_component(
        components: list[tuple[str, int, int, float]],
        direction: str,
        i_nb: int,
        j_nb: int,
        coefficient: float,
    ) -> None:
        if 0 <= i_nb < len(rho) and 0 <= j_nb < len(z) and (not strict[i_nb, j_nb]) and coefficient > 0.0:
            components.append((direction, i_nb, j_nb, coefficient))

    point_rows = []
    component_rows = []
    trace_points: dict[tuple[int, int], dict[str, float | str | int]] = {}
    edge_totals: dict[str, dict[str, float | int]] = {}
    all_trace_floor_valid = True
    pointwise_old_b_floor_valid = True

    for edge in edge_order:
        global_idx = np.flatnonzero(categories[edge].ravel())
        global_idx = sorted(
            global_idx,
            key=lambda idx_flat: (
                z[np.unravel_index(idx_flat, strict.shape)[1]],
                rho[np.unravel_index(idx_flat, strict.shape)[0]],
            ),
        )
        edge_exact_product = 0.0
        edge_old_cert_product = 0.0
        edge_barrier_product = 0.0
        edge_flux_product = 0.0
        edge_b_exact = 0.0
        edge_b_floor = 0.0
        edge_flux_floor = 0.0
        edge_barrier_flux = 0.0
        edge_old_b_floor_failures = 0

        for point_id, idx_flat in enumerate(global_idx, start=1):
            i_rho, i_z = np.unravel_index(idx_flat, strict.shape)
            local_idx = local_of_global[idx_flat]
            components: list[tuple[str, int, int, float]] = []
            add_component(
                components,
                'rho_plus',
                i_rho + 1,
                i_z,
                ((i_rho + 1) * level.dr) / (level.dr * level.dr),
            )
            add_component(
                components,
                'rho_minus',
                i_rho - 1,
                i_z,
                (i_rho * level.dr) / (level.dr * level.dr),
            )
            add_component(
                components,
                'z_plus',
                i_rho,
                i_z + 1,
                rho[i_rho] / (level.dz * level.dz),
            )
            add_component(
                components,
                'z_minus',
                i_rho,
                i_z - 1,
                rho[i_rho] / (level.dz * level.dz),
            )

            b_exact = float(boundary_defect[local_idx])
            b_floor = floor_sig(b_exact, B_SIG_FIGS)
            p_exact = float(poisson_weight[local_idx])
            p_floor = floor_sig(p_exact, P_SIG_FIGS)
            flux_exact = 0.0
            flux_floor = 0.0
            flux_barrier = 0.0

            for direction, i_nb, j_nb, coefficient in components:
                phi_exact = float(phi_L[i_nb, j_nb])
                phi_floor = floor_sig(phi_exact, PHI_SIG_FIGS)
                phi_barrier = PHI_BARRIER_THETA * phi_exact
                component_exact = coefficient * phi_exact
                component_barrier = coefficient * phi_barrier
                component_floor = coefficient * phi_floor
                flux_exact += component_exact
                flux_barrier += component_barrier
                flux_floor += component_floor
                all_trace_floor_valid = all_trace_floor_valid and (phi_barrier + 1e-15 >= phi_floor)
                trace_points.setdefault(
                    (i_nb, j_nb),
                    {
                        'rho': float(rho[i_nb]),
                        'z_rel': float(z[j_nb] - D / 2.0),
                        'phi_exact': phi_exact,
                        'phi_barrier': phi_barrier,
                        'phi_floor': phi_floor,
                        'floor_over_exact': phi_floor / max(phi_exact, 1e-300),
                        'barrier_minus_floor': phi_barrier - phi_floor,
                    },
                )
                component_rows.append(
                    {
                        'edge': edge,
                        'edge_label': labels[edge],
                        'point_id': point_id,
                        'direction': direction,
                        'rho_q': float(rho[i_rho]),
                        'z_rel_q': float(z[i_z] - D / 2.0),
                        'rho_out': float(rho[i_nb]),
                        'z_rel_out': float(z[j_nb] - D / 2.0),
                        'coefficient': float(coefficient),
                        'phi_exact': phi_exact,
                        'phi_barrier': phi_barrier,
                        'phi_floor': phi_floor,
                        'barrier_minus_floor': phi_barrier - phi_floor,
                        'component_exact': component_exact,
                        'component_barrier': component_barrier,
                        'component_certified': component_floor,
                        'component_barrier_minus_certified': component_barrier - component_floor,
                    }
                )

            exact_product = b_exact * p_exact
            old_cert_product = b_floor * p_floor
            barrier_product = flux_barrier * p_floor
            flux_product = flux_floor * p_floor
            point_old_valid = flux_floor + 1e-12 >= b_floor
            pointwise_old_b_floor_valid = pointwise_old_b_floor_valid and point_old_valid
            edge_old_b_floor_failures += 0 if point_old_valid else 1

            edge_exact_product += exact_product
            edge_old_cert_product += old_cert_product
            edge_barrier_product += barrier_product
            edge_flux_product += flux_product
            edge_b_exact += b_exact
            edge_b_floor += b_floor
            edge_flux_floor += flux_floor
            edge_barrier_flux += flux_barrier

            point_rows.append(
                {
                    'edge': edge,
                    'edge_label': labels[edge],
                    'point_id': point_id,
                    'rho': float(rho[i_rho]),
                    'z_rel': float(z[i_z] - D / 2.0),
                    'missing_component_count': len(components),
                    'exact_boundary_defect': b_exact,
                    'certified_boundary_defect_floor': b_floor,
                    'barrier_flux_floor': flux_barrier,
                    'certified_flux_floor': flux_floor,
                    'flux_floor_minus_old_boundary_floor': flux_floor - b_floor,
                    'exact_poisson_weight': p_exact,
                    'certified_poisson_weight_floor': p_floor,
                    'exact_product': exact_product,
                    'old_certified_product_floor': old_cert_product,
                    'barrier_flux_product_with_p_floor': barrier_product,
                    'direct_flux_product_with_p_floor': flux_product,
                    'old_point_b_floor_still_valid': point_old_valid,
                }
            )

        edge_totals[edge] = {
            'exact_boundary_defect_sum': edge_b_exact,
            'old_certified_boundary_defect_sum': edge_b_floor,
            'barrier_flux_sum': edge_barrier_flux,
            'direct_flux_floor_sum': edge_flux_floor,
            'exact_product': edge_exact_product,
            'old_certified_product_floor': edge_old_cert_product,
            'barrier_flux_product_with_p_floor': edge_barrier_product,
            'direct_flux_product_with_p_floor': edge_flux_product,
            'old_b_floor_failure_count': edge_old_b_floor_failures,
        }

    trace_order = sorted(trace_points, key=lambda ij: (rho[ij[0]], z[ij[1]]))
    trace_flat = np.array([flat_index(i, j) for i, j in trace_order], dtype=int)
    k_trace = klap[trace_flat][:, trace_flat].tocsr()
    m_trace = m_mat[trace_flat][:, trace_flat].tocsr()
    phi_trace = np.array([float(phi_L[i, j]) for i, j in trace_order])
    rhs_trace = np.array([float(rhs0[i, j]) for i, j in trace_order])
    phi_barrier_trace = PHI_BARRIER_THETA * phi_trace
    parent_outside_flux = k_trace @ phi_trace - m_trace @ rhs_trace
    comparison_residual = k_trace @ (phi_trace - phi_barrier_trace)
    comparison_expected = (1.0 - PHI_BARRIER_THETA) * (m_trace @ rhs_trace + parent_outside_flux)
    comparison_agreement = comparison_residual - comparison_expected

    coo = k_trace.tocoo()
    offdiag = coo.data[coo.row != coo.col]
    diag = k_trace.diagonal()
    row_abs_offdiag = np.asarray(abs(k_trace).sum(axis=1)).ravel() - np.abs(diag)
    diagonal_margin = diag - row_abs_offdiag

    trace_set = set(trace_order)
    halo = set()
    for i, j in trace_order:
        for i_nb, j_nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= i_nb < len(rho) and 0 <= j_nb < len(z) and (i_nb, j_nb) not in trace_set:
                halo.add((i_nb, j_nb))
    halo_phi = np.array([float(phi_L[i, j]) for i, j in sorted(halo)]) if halo else np.array([float('nan')])

    trace_rows = []
    for trace_id, (i, j) in enumerate(trace_order, start=1):
        data = trace_points[(i, j)]
        trace_rows.append(
            {
                'trace_id': trace_id,
                'rho': data['rho'],
                'z_rel': data['z_rel'],
                'phi_exact': data['phi_exact'],
                'phi_barrier': data['phi_barrier'],
                'phi_floor': data['phi_floor'],
                'floor_over_exact': data['floor_over_exact'],
                'barrier_minus_floor': data['barrier_minus_floor'],
                'rhs_trace': rhs_trace[trace_id - 1],
                'parent_outside_flux': parent_outside_flux[trace_id - 1],
                'comparison_residual': comparison_residual[trace_id - 1],
                'comparison_expected': comparison_expected[trace_id - 1],
                'comparison_agreement_abs': abs(comparison_agreement[trace_id - 1]),
            }
        )

    edge_rows = []
    cumulative_old = 0.0
    cumulative_direct = 0.0
    cumulative_barrier = 0.0
    for edge in edge_order:
        row = edge_totals[edge]
        cumulative_old += float(row['old_certified_product_floor'])
        cumulative_barrier += float(row['barrier_flux_product_with_p_floor'])
        cumulative_direct += float(row['direct_flux_product_with_p_floor'])
        edge_rows.append(
            {
                'edge': edge,
                'edge_label': labels[edge],
                'exact_boundary_defect_sum': row['exact_boundary_defect_sum'],
                'old_certified_boundary_defect_sum': row['old_certified_boundary_defect_sum'],
                'barrier_flux_sum': row['barrier_flux_sum'],
                'direct_flux_floor_sum': row['direct_flux_floor_sum'],
                'direct_flux_floor_minus_old_boundary_floor_sum': float(row['direct_flux_floor_sum'])
                - float(row['old_certified_boundary_defect_sum']),
                'exact_product': row['exact_product'],
                'old_certified_product_floor': row['old_certified_product_floor'],
                'barrier_flux_product_with_p_floor': row['barrier_flux_product_with_p_floor'],
                'direct_flux_product_with_p_floor': row['direct_flux_product_with_p_floor'],
                'old_b_floor_failure_count': row['old_b_floor_failure_count'],
                'cumulative_old_certified_product': cumulative_old,
                'cumulative_barrier_product': cumulative_barrier,
                'cumulative_direct_flux_product': cumulative_direct,
                'cumulative_direct_over_global_after_source': cumulative_direct
                / max(target_global_after_source, 1e-300),
                'cumulative_direct_over_local_after_source': cumulative_direct
                / max(target_local_after_source, 1e-300),
            }
        )

    exact_edge_product_sum = sum(float(row['exact_product']) for row in edge_totals.values())
    old_certified_edge_sum = sum(float(row['old_certified_product_floor']) for row in edge_totals.values())
    barrier_edge_sum = sum(float(row['barrier_flux_product_with_p_floor']) for row in edge_totals.values())
    direct_flux_edge_sum = sum(float(row['direct_flux_product_with_p_floor']) for row in edge_totals.values())
    exact_b_sum = sum(float(row['exact_boundary_defect_sum']) for row in edge_totals.values())
    b_floor_sum = sum(float(row['old_certified_boundary_defect_sum']) for row in edge_totals.values())
    barrier_flux_sum = sum(float(row['barrier_flux_sum']) for row in edge_totals.values())
    direct_flux_sum = sum(float(row['direct_flux_floor_sum']) for row in edge_totals.values())
    direct_total_with_source = direct_flux_edge_sum + source
    direct_outer_bottom = (
        float(edge_totals['outer_rho']['direct_flux_product_with_p_floor'])
        + float(edge_totals['bottom']['direct_flux_product_with_p_floor'])
    )
    direct_route_valid = (
        all_trace_floor_valid
        and direct_flux_edge_sum >= target_global_after_source
        and direct_total_with_source >= required_global
        and direct_outer_bottom >= target_local_after_source
    )
    m_matrix_valid = (
        float(np.min(diag)) > 0.0
        and float(np.max(offdiag)) <= 0.0
        and float(np.min(diagonal_margin)) > 0.0
    )
    comparison_valid = (
        m_matrix_valid
        and float(np.min(rhs_trace)) > 0.0
        and float(np.min(parent_outside_flux)) > 0.0
        and float(np.min(comparison_expected)) > 0.0
        and float(np.min(halo_phi)) > 0.0
    )

    summary = {
        'D': D,
        'rho_anchor': rho_anchor,
        'zeta_anchor': zeta_anchor,
        'strict_cap': h_star,
        'boundary_defect_sig_figs': B_SIG_FIGS,
        'exterior_phi_sig_figs': PHI_SIG_FIGS,
        'poisson_weight_sig_figs': P_SIG_FIGS,
        'phi_barrier_theta': PHI_BARRIER_THETA,
        'required_global': required_global,
        'required_local': required_local,
        'source_potential_mass': source,
        'target_global_after_source': target_global_after_source,
        'target_local_after_source': target_local_after_source,
        'edge_point_count': len(point_rows),
        'exterior_trace_point_count': len(trace_rows),
        'missing_flux_component_count': len(component_rows),
        'trace_halo_point_count': len(halo),
        'min_exterior_phi_exact': float(np.min(phi_trace)),
        'max_exterior_phi_exact': float(np.max(phi_trace)),
        'max_phi_floor_over_exact': float(np.max(np.array([row['floor_over_exact'] for row in trace_rows]))),
        'min_phi_barrier_minus_floor': float(np.min(np.array([row['barrier_minus_floor'] for row in trace_rows]))),
        'min_phi_exact_minus_barrier': float(np.min(phi_trace - phi_barrier_trace)),
        'min_trace_rhs': float(np.min(rhs_trace)),
        'max_trace_rhs': float(np.max(rhs_trace)),
        'min_parent_outside_flux': float(np.min(parent_outside_flux)),
        'max_parent_outside_flux': float(np.max(parent_outside_flux)),
        'min_trace_halo_phi': float(np.min(halo_phi)),
        'min_trace_diag': float(np.min(diag)),
        'max_trace_offdiag': float(np.max(offdiag)),
        'min_trace_diagonal_margin': float(np.min(diagonal_margin)),
        'min_comparison_expected': float(np.min(comparison_expected)),
        'max_comparison_expected': float(np.max(comparison_expected)),
        'max_comparison_agreement_abs': float(np.max(np.abs(comparison_agreement))),
        'exact_boundary_defect_sum': exact_b_sum,
        'old_certified_boundary_defect_floor_sum': b_floor_sum,
        'barrier_flux_sum': barrier_flux_sum,
        'direct_flux_floor_sum': direct_flux_sum,
        'direct_flux_minus_old_boundary_floor_sum': direct_flux_sum - b_floor_sum,
        'old_pointwise_b_floor_fail_count': int(
            sum(int(row['old_b_floor_failure_count']) for row in edge_totals.values())
        ),
        'old_pointwise_b_floor_valid_under_4sig_phi': pointwise_old_b_floor_valid,
        'exact_edge_product_sum': exact_edge_product_sum,
        'old_certified_edge_sum': old_certified_edge_sum,
        'barrier_flux_product_with_p_floor_sum': barrier_edge_sum,
        'direct_flux_product_with_p_floor_sum': direct_flux_edge_sum,
        'direct_flux_product_over_global_after_source': direct_flux_edge_sum
        / max(target_global_after_source, 1e-300),
        'direct_flux_product_over_local_after_source': direct_flux_edge_sum
        / max(target_local_after_source, 1e-300),
        'direct_flux_gap_over_global_after_source': direct_flux_edge_sum - target_global_after_source,
        'direct_total_with_source': direct_total_with_source,
        'direct_total_over_required_global': direct_total_with_source / max(required_global, 1e-300),
        'direct_total_over_required_local': direct_total_with_source / max(required_local, 1e-300),
        'direct_outer_bottom_product': direct_outer_bottom,
        'direct_outer_bottom_over_local_after_source': direct_outer_bottom
        / max(target_local_after_source, 1e-300),
        'trace_floor_certificate_valid': all_trace_floor_valid,
        'trace_comparison_certificate_valid': comparison_valid,
        'direct_flux_route_valid': direct_route_valid,
        'certificate_route': 'four_sig_exterior_phi_barrier_direct_flux_product',
    }

    summary_path = (
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_exterior_phi_barrier_source_summary.csv'
    )
    edge_path = (
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_exterior_phi_barrier_source_edges.csv'
    )
    detail_path = (
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_exterior_phi_barrier_source_detail.csv'
    )
    component_path = (
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_exterior_phi_barrier_source_components.csv'
    )
    trace_path = (
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_exterior_phi_barrier_source_trace.csv'
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
    with component_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0].keys()))
        writer.writeheader()
        for row in component_rows:
            writer.writerow(row)
    with trace_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        for row in trace_rows:
            writer.writerow(row)

    print(summary_path)
    print(edge_path)
    print(detail_path)
    print(component_path)
    print(trace_path)


if __name__ == '__main__':
    main()
