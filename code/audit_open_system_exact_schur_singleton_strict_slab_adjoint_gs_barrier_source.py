#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import diags, tril, triu
from scipy.sparse.linalg import splu, spsolve

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from extract_chi_localized_2d import Level, PhysicalParams, build_generalized_operator

OUT_DIR = ROOT / 'output' / 'chi_open_system'
PRODUCT_SWEEPS = 548
FLOOR_SWEEPS = 859


def main() -> None:
    strict_summary = next(
        csv.DictReader(
            (OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_strict_slab_source_summary.csv').open()
        )
    )
    exterior_summary = next(
        csv.DictReader(
            (
                OUT_DIR
                / 'chi_open_system_exact_schur_singleton_strict_slab_exterior_phi_barrier_source_summary.csv'
            ).open()
        )
    )
    exterior_detail = list(
        csv.DictReader(
            (
                OUT_DIR
                / 'chi_open_system_exact_schur_singleton_strict_slab_exterior_phi_barrier_source_detail.csv'
            ).open()
        )
    )

    D = float(strict_summary['D'])
    rho_anchor = float(strict_summary['rho_anchor'])
    zeta_anchor = float(strict_summary['zeta_anchor'])
    h_star = float(strict_summary['strict_cap'])
    required_global = float(exterior_summary['required_global'])
    required_local = float(exterior_summary['required_local'])
    source = float(exterior_summary['source_potential_mass'])
    target_global_after_source = float(exterior_summary['target_global_after_source'])
    target_local_after_source = float(exterior_summary['target_local_after_source'])

    p = PhysicalParams()
    level = Level('fine', dr=0.06, dz=0.03)
    z_max = D / 2.0 + 6.0
    rho, z, rr, zz, uu, k_mat, _ = build_generalized_operator(
        D=D,
        p=p,
        rho_max=3.0,
        z_max=z_max,
        dr=level.dr,
        dz=level.dz,
    )
    klap = (k_mat - diags((rr * uu).ravel(), 0, format='csr')).tocsr()

    plus = (rr <= rho_anchor) & (np.abs(zz - D / 2.0) <= zeta_anchor)
    strict = plus & ((zz - D / 2.0) <= -h_star)
    flat = np.flatnonzero(strict.ravel())
    local_of_global = -np.ones(strict.size, dtype=int)
    local_of_global[flat] = np.arange(len(flat))

    ksub = klap[flat][:, flat].tocsr()
    adjoint = ksub.T.tocsr()
    weight = (4.0 * np.pi * rr * level.dr * level.dz).ravel()[flat]
    exact_poisson = spsolve(adjoint, weight)

    diag = adjoint.diagonal()
    b_mat = (-adjoint).tocsr()
    b_mat.setdiag(0.0)
    lower = tril(b_mat, k=-1, format='csc')
    upper = triu(b_mat, k=1, format='csr')
    gs_matrix = (diags(diag, 0, format='csr') - lower).tocsc()
    gs_lu = splu(gs_matrix)

    edge_order = ['outer_rho', 'bottom', 'corner_rb', 'corner_lb']
    edge_labels = {
        'outer_rho': 'out',
        'bottom': 'bot',
        'corner_rb': 'rb',
        'corner_lb': 'lb',
    }

    point_specs = []
    for row in exterior_detail:
        i_rho = int(round(float(row['rho']) / level.dr - 0.5))
        i_z = int(round((float(row['z_rel']) + D / 2.0 + z_max) / level.dz - 0.5))
        global_idx = np.ravel_multi_index((i_rho, i_z), strict.shape)
        local_idx = int(local_of_global[global_idx])
        if local_idx < 0:
            raise RuntimeError(f'edge point outside strict slab: {row}')
        point_specs.append(
            {
                'edge': row['edge'],
                'edge_label': row['edge_label'],
                'point_id': int(row['point_id']),
                'rho': float(row['rho']),
                'z_rel': float(row['z_rel']),
                'local_idx': local_idx,
                'direct_flux_floor': float(row['certified_flux_floor']),
                'poisson_floor': float(row['certified_poisson_weight_floor']),
                'poisson_exact_from_exterior_detail': float(row['exact_poisson_weight']),
            }
        )

    selected_sweeps = {
        0,
        1,
        2,
        3,
        5,
        10,
        20,
        50,
        100,
        200,
        300,
        384,
        400,
        500,
        PRODUCT_SWEEPS,
        600,
        700,
        800,
        FLOOR_SWEEPS,
    }
    selected_sweeps = {n for n in selected_sweeps if 0 <= n <= FLOOR_SWEEPS}

    def product_metrics(vec: np.ndarray) -> dict[str, float]:
        total = 0.0
        outer_bottom = 0.0
        min_floor_ratio = float('inf')
        min_exact_ratio = float('inf')
        for spec in point_specs:
            value = float(vec[spec['local_idx']])
            product = spec['direct_flux_floor'] * value
            total += product
            if spec['edge'] in {'outer_rho', 'bottom'}:
                outer_bottom += product
            min_floor_ratio = min(min_floor_ratio, value / max(spec['poisson_floor'], 1e-300))
            exact_value = float(exact_poisson[spec['local_idx']])
            min_exact_ratio = min(min_exact_ratio, value / max(exact_value, 1e-300))
        return {
            'direct_flux_product': total,
            'outer_bottom_product': outer_bottom,
            'direct_flux_product_over_global_after_source': total / max(target_global_after_source, 1e-300),
            'outer_bottom_over_local_after_source': outer_bottom / max(target_local_after_source, 1e-300),
            'total_with_source_over_required_global': (total + source) / max(required_global, 1e-300),
            'min_point_over_4sig_p_floor': min_floor_ratio,
            'min_point_over_exact_poisson': min_exact_ratio,
        }

    iter_rows = []
    current = np.zeros(adjoint.shape[0])
    previous = current.copy()
    product_vec = None
    floor_vec = None
    product_residual = None
    floor_residual = None
    product_increment = None
    floor_increment = None

    metrics = product_metrics(current)
    if 0 in selected_sweeps:
        iter_rows.append(
            {
                'sweep': 0,
                **metrics,
                'min_increment': 0.0,
                'min_subsolution_residual': 0.0,
                'max_subsolution_residual': 0.0,
                'residual_identity_abs_max': 0.0,
            }
        )

    for sweep in range(1, FLOOR_SWEEPS + 1):
        previous = current
        current = gs_lu.solve(weight + upper @ previous)
        increment = current - previous
        residual = weight - adjoint @ current
        if sweep in selected_sweeps:
            metrics = product_metrics(current)
            iter_rows.append(
                {
                    'sweep': sweep,
                    **metrics,
                    'min_increment': float(np.min(increment)),
                    'min_subsolution_residual': float(np.min(residual)),
                    'max_subsolution_residual': float(np.max(residual)),
                    'residual_identity_abs_max': float(np.max(np.abs(residual - upper @ increment))),
                }
            )
        if sweep == PRODUCT_SWEEPS:
            product_vec = current.copy()
            product_residual = residual.copy()
            product_increment = increment.copy()
        if sweep == FLOOR_SWEEPS:
            floor_vec = current.copy()
            floor_residual = residual.copy()
            floor_increment = increment.copy()

    if product_vec is None or floor_vec is None or product_residual is None or floor_residual is None:
        raise RuntimeError('missing saved Gauss-Seidel sweep')

    product_metrics_final = product_metrics(product_vec)
    floor_metrics_final = product_metrics(floor_vec)

    detail_rows = []
    for spec in point_specs:
        local_idx = spec['local_idx']
        p_product = float(product_vec[local_idx])
        p_floor_sweep = float(floor_vec[local_idx])
        p_exact = float(exact_poisson[local_idx])
        detail_rows.append(
            {
                'edge': spec['edge'],
                'edge_label': spec['edge_label'],
                'point_id': spec['point_id'],
                'rho': spec['rho'],
                'z_rel': spec['z_rel'],
                'direct_flux_floor': spec['direct_flux_floor'],
                'poisson_exact': p_exact,
                'poisson_exact_resid_vs_exterior_detail_abs': abs(
                    p_exact - spec['poisson_exact_from_exterior_detail']
                ),
                'poisson_floor_4sig': spec['poisson_floor'],
                'gs_product_sweep_poisson': p_product,
                'gs_floor_sweep_poisson': p_floor_sweep,
                'product_sweep_over_4sig_floor': p_product / max(spec['poisson_floor'], 1e-300),
                'floor_sweep_over_4sig_floor': p_floor_sweep / max(spec['poisson_floor'], 1e-300),
                'product_sweep_over_exact': p_product / max(p_exact, 1e-300),
                'floor_sweep_over_exact': p_floor_sweep / max(p_exact, 1e-300),
                'product_sweep_flux_product': spec['direct_flux_floor'] * p_product,
                'floor_sweep_flux_product': spec['direct_flux_floor'] * p_floor_sweep,
                'old_4sig_floor_flux_product': spec['direct_flux_floor'] * spec['poisson_floor'],
                'product_sweep_point_floor_valid': p_product + 1e-15 >= spec['poisson_floor'],
                'floor_sweep_point_floor_valid': p_floor_sweep + 1e-15 >= spec['poisson_floor'],
            }
        )

    edge_rows = []
    cumulative_product = 0.0
    cumulative_floor = 0.0
    cumulative_old = 0.0
    for edge in edge_order:
        rows = [row for row in detail_rows if row['edge'] == edge]
        product_sum = sum(row['product_sweep_flux_product'] for row in rows)
        floor_sum = sum(row['floor_sweep_flux_product'] for row in rows)
        old_sum = sum(row['old_4sig_floor_flux_product'] for row in rows)
        exact_sum = sum(row['direct_flux_floor'] * row['poisson_exact'] for row in rows)
        cumulative_product += product_sum
        cumulative_floor += floor_sum
        cumulative_old += old_sum
        edge_rows.append(
            {
                'edge': edge,
                'edge_label': edge_labels[edge],
                'point_count': len(rows),
                'direct_flux_sum': sum(row['direct_flux_floor'] for row in rows),
                'exact_flux_product': exact_sum,
                'gs_product_sweep_flux_product': product_sum,
                'gs_floor_sweep_flux_product': floor_sum,
                'old_4sig_floor_flux_product': old_sum,
                'cumulative_product_sweep_flux_product': cumulative_product,
                'cumulative_floor_sweep_flux_product': cumulative_floor,
                'cumulative_old_4sig_floor_flux_product': cumulative_old,
                'cumulative_product_over_global_after_source': cumulative_product
                / max(target_global_after_source, 1e-300),
                'cumulative_product_over_local_after_source': cumulative_product
                / max(target_local_after_source, 1e-300),
            }
        )

    coo = adjoint.tocoo()
    offdiag = coo.data[coo.row != coo.col]
    b_coo = b_mat.tocoo()
    b_offdiag = b_coo.data[b_coo.row != b_coo.col]
    lower_coo = lower.tocoo()
    upper_coo = upper.tocoo()
    gs_diag = gs_matrix.diagonal()

    summary = {
        'D': D,
        'rho_anchor': rho_anchor,
        'zeta_anchor': zeta_anchor,
        'strict_cap': h_star,
        'product_sweeps': PRODUCT_SWEEPS,
        'floor_sweeps': FLOOR_SWEEPS,
        'required_global': required_global,
        'required_local': required_local,
        'source_potential_mass': source,
        'target_global_after_source': target_global_after_source,
        'target_local_after_source': target_local_after_source,
        'strict_slab_point_count': int(adjoint.shape[0]),
        'edge_point_count': len(point_specs),
        'min_adjoint_diag': float(np.min(diag)),
        'max_adjoint_offdiag': float(np.max(offdiag)),
        'min_B_offdiag': float(np.min(b_offdiag)),
        'max_B_offdiag': float(np.max(b_offdiag)),
        'lower_nonzero_count': int(lower.nnz),
        'upper_nonzero_count': int(upper.nnz),
        'min_gs_diag': float(np.min(gs_diag)),
        'product_sweep_min_increment': float(np.min(product_increment)),
        'product_sweep_min_subsolution_residual': float(np.min(product_residual)),
        'product_sweep_max_subsolution_residual': float(np.max(product_residual)),
        'product_sweep_residual_identity_abs_max': float(
            np.max(np.abs(product_residual - upper @ product_increment))
        ),
        'floor_sweep_min_increment': float(np.min(floor_increment)),
        'floor_sweep_min_subsolution_residual': float(np.min(floor_residual)),
        'floor_sweep_max_subsolution_residual': float(np.max(floor_residual)),
        'floor_sweep_residual_identity_abs_max': float(
            np.max(np.abs(floor_residual - upper @ floor_increment))
        ),
        'product_sweep_direct_flux_product': product_metrics_final['direct_flux_product'],
        'product_sweep_direct_flux_product_over_global_after_source': product_metrics_final[
            'direct_flux_product_over_global_after_source'
        ],
        'product_sweep_direct_flux_gap_over_global_after_source': product_metrics_final[
            'direct_flux_product'
        ]
        - target_global_after_source,
        'product_sweep_total_with_source': product_metrics_final['direct_flux_product'] + source,
        'product_sweep_total_over_required_global': product_metrics_final[
            'total_with_source_over_required_global'
        ],
        'product_sweep_outer_bottom_product': product_metrics_final['outer_bottom_product'],
        'product_sweep_outer_bottom_over_local_after_source': product_metrics_final[
            'outer_bottom_over_local_after_source'
        ],
        'product_sweep_outer_bottom_gap_over_local_after_source': product_metrics_final[
            'outer_bottom_product'
        ]
        - target_local_after_source,
        'product_sweep_min_point_over_4sig_p_floor': product_metrics_final['min_point_over_4sig_p_floor'],
        'product_sweep_min_point_over_exact_poisson': product_metrics_final['min_point_over_exact_poisson'],
        'floor_sweep_direct_flux_product': floor_metrics_final['direct_flux_product'],
        'floor_sweep_direct_flux_product_over_global_after_source': floor_metrics_final[
            'direct_flux_product_over_global_after_source'
        ],
        'floor_sweep_total_over_required_global': floor_metrics_final['total_with_source_over_required_global'],
        'floor_sweep_min_point_over_4sig_p_floor': floor_metrics_final['min_point_over_4sig_p_floor'],
        'floor_sweep_min_point_over_exact_poisson': floor_metrics_final['min_point_over_exact_poisson'],
        'old_4sig_flux_product': sum(row['old_4sig_floor_flux_product'] for row in detail_rows),
        'old_4sig_flux_product_over_global_after_source': sum(
            row['old_4sig_floor_flux_product'] for row in detail_rows
        )
        / max(target_global_after_source, 1e-300),
        'product_sweep_direct_route_valid': product_metrics_final['direct_flux_product']
        >= target_global_after_source
        and product_metrics_final['outer_bottom_product'] >= target_local_after_source
        and float(np.min(product_increment)) >= -1e-14
        and float(np.min(product_residual)) >= -1e-14,
        'floor_sweep_point_floor_certificate_valid': all(
            row['floor_sweep_point_floor_valid'] for row in detail_rows
        )
        and float(np.min(floor_increment)) >= -1e-14
        and float(np.min(floor_residual)) >= -1e-14,
        'certificate_route': 'monotone_gauss_seidel_adjoint_mmatrix_lower_iterates',
    }

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_adjoint_gs_barrier_source_summary.csv'
    edge_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_adjoint_gs_barrier_source_edges.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_adjoint_gs_barrier_source_detail.csv'
    iter_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_adjoint_gs_barrier_source_iterations.csv'

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
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)
    with iter_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(iter_rows[0].keys()))
        writer.writeheader()
        for row in iter_rows:
            writer.writerow(row)

    print(summary_path)
    print(edge_path)
    print(detail_path)
    print(iter_path)


if __name__ == '__main__':
    main()
