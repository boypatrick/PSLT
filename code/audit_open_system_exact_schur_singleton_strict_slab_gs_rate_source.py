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
RATE_CHECKPOINTS = (384, 500, PRODUCT_SWEEPS, 700, 800, FLOOR_SWEEPS)


def read_one(path: Path) -> dict[str, str]:
    return next(csv.DictReader(path.open()))


def main() -> None:
    strict_summary = read_one(
        OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_strict_slab_source_summary.csv'
    )
    exterior_summary = read_one(
        OUT_DIR
        / 'chi_open_system_exact_schur_singleton_strict_slab_exterior_phi_barrier_source_summary.csv'
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
                'local_idx': local_idx,
                'direct_flux_floor': float(row['certified_flux_floor']),
                'poisson_floor': float(row['certified_poisson_weight_floor']),
            }
        )

    def flux_product(vec: np.ndarray) -> float:
        return float(sum(spec['direct_flux_floor'] * vec[spec['local_idx']] for spec in point_specs))

    def outer_bottom_product(vec: np.ndarray) -> float:
        return float(
            sum(
                spec['direct_flux_floor'] * vec[spec['local_idx']]
                for spec in point_specs
                if spec['edge'] in {'outer_rho', 'bottom'}
            )
        )

    def min_point_floor_ratio(vec: np.ndarray) -> float:
        return float(
            min(vec[spec['local_idx']] / max(spec['poisson_floor'], 1e-300) for spec in point_specs)
        )

    exact_flux_product = flux_product(exact_poisson)
    exact_outer_bottom_product = outer_bottom_product(exact_poisson)

    max_sweep = max(RATE_CHECKPOINTS) + 1
    current = np.zeros(adjoint.shape[0])
    sweeps: dict[int, np.ndarray] = {0: current.copy()}
    increments: dict[int, np.ndarray] = {}
    first_global_sweep = None
    first_local_sweep = None
    first_both_sweep = None
    first_point_floor_sweep = None

    for sweep in range(1, max_sweep + 1):
        previous = current
        current = gs_lu.solve(weight + upper @ previous)
        increment = current - previous
        sweeps[sweep] = current.copy()
        increments[sweep] = increment.copy()

        global_closed = flux_product(current) >= target_global_after_source
        local_closed = outer_bottom_product(current) >= target_local_after_source
        point_floor_closed = min_point_floor_ratio(current) >= 1.0
        if first_global_sweep is None and global_closed:
            first_global_sweep = sweep
        if first_local_sweep is None and local_closed:
            first_local_sweep = sweep
        if first_both_sweep is None and global_closed and local_closed:
            first_both_sweep = sweep
        if first_point_floor_sweep is None and point_floor_closed:
            first_point_floor_sweep = sweep

    checkpoint_rows = []
    for sweep in RATE_CHECKPOINTS:
        vec = sweeps[sweep]
        inc = increments[sweep]
        next_inc = increments[sweep + 1]
        ratios = next_inc / np.maximum(inc, 1e-300)
        alpha = float(np.max(ratios))
        beta = float(np.min(ratios))
        tail_factor = alpha / (1.0 - alpha) if alpha < 1.0 else float('inf')
        product = flux_product(vec)
        local_product = outer_bottom_product(vec)
        product_increment = flux_product(inc)
        local_increment = outer_bottom_product(inc)
        product_tail_bound = tail_factor * product_increment
        local_tail_bound = tail_factor * local_increment
        exact_product_gap = exact_flux_product - product
        exact_local_gap = exact_outer_bottom_product - local_product
        checkpoint_rows.append(
            {
                'sweep': sweep,
                'alpha_component_upper': alpha,
                'beta_component_lower': beta,
                'tail_factor_alpha_over_1_minus_alpha': tail_factor,
                'min_increment': float(np.min(inc)),
                'max_increment': float(np.max(inc)),
                'flux_product': product,
                'flux_product_over_global_after_source': product / max(target_global_after_source, 1e-300),
                'outer_bottom_product': local_product,
                'outer_bottom_over_local_after_source': local_product / max(target_local_after_source, 1e-300),
                'point_floor_min_ratio': min_point_floor_ratio(vec),
                'flux_product_increment': product_increment,
                'outer_bottom_increment': local_increment,
                'flux_product_tail_bound': product_tail_bound,
                'outer_bottom_tail_bound': local_tail_bound,
                'exact_flux_product_gap': exact_product_gap,
                'exact_outer_bottom_gap': exact_local_gap,
                'tail_bound_covers_exact_flux_gap': product_tail_bound + 1e-14 >= exact_product_gap,
                'tail_bound_covers_exact_outer_bottom_gap': local_tail_bound + 1e-14 >= exact_local_gap,
            }
        )

    product_row = next(row for row in checkpoint_rows if row['sweep'] == PRODUCT_SWEEPS)
    floor_row = next(row for row in checkpoint_rows if row['sweep'] == FLOOR_SWEEPS)

    summary = {
        'D': D,
        'rho_anchor': rho_anchor,
        'zeta_anchor': zeta_anchor,
        'strict_cap': h_star,
        'strict_slab_point_count': int(adjoint.shape[0]),
        'edge_point_count': len(point_specs),
        'product_sweeps': PRODUCT_SWEEPS,
        'floor_sweeps': FLOOR_SWEEPS,
        'required_global': required_global,
        'required_local': required_local,
        'source_potential_mass': source,
        'target_global_after_source': target_global_after_source,
        'target_local_after_source': target_local_after_source,
        'first_global_sweep': first_global_sweep,
        'first_local_sweep': first_local_sweep,
        'first_both_sweep': first_both_sweep,
        'first_point_floor_sweep': first_point_floor_sweep,
        'exact_flux_product': exact_flux_product,
        'exact_flux_product_over_global_after_source': exact_flux_product
        / max(target_global_after_source, 1e-300),
        'exact_total_with_source_over_required_global': (exact_flux_product + source)
        / max(required_global, 1e-300),
        'exact_outer_bottom_product': exact_outer_bottom_product,
        'exact_outer_bottom_over_local_after_source': exact_outer_bottom_product
        / max(target_local_after_source, 1e-300),
        'product_sweep_alpha': product_row['alpha_component_upper'],
        'product_sweep_tail_factor': product_row['tail_factor_alpha_over_1_minus_alpha'],
        'product_sweep_flux_tail_bound': product_row['flux_product_tail_bound'],
        'product_sweep_exact_flux_gap': product_row['exact_flux_product_gap'],
        'product_sweep_tail_bound_valid': product_row['tail_bound_covers_exact_flux_gap'],
        'floor_sweep_alpha': floor_row['alpha_component_upper'],
        'floor_sweep_tail_factor': floor_row['tail_factor_alpha_over_1_minus_alpha'],
        'floor_sweep_flux_tail_bound': floor_row['flux_product_tail_bound'],
        'floor_sweep_exact_flux_gap': floor_row['exact_flux_product_gap'],
        'floor_sweep_tail_bound_valid': floor_row['tail_bound_covers_exact_flux_gap'],
        'product_sweep_direct_route_valid': product_row['flux_product'] >= target_global_after_source
        and product_row['outer_bottom_product'] >= target_local_after_source,
        'floor_sweep_point_floor_valid': floor_row['point_floor_min_ratio'] >= 1.0,
        'all_checkpoint_tail_bounds_valid': all(
            row['alpha_component_upper'] < 1.0
            and row['tail_bound_covers_exact_flux_gap']
            and row['tail_bound_covers_exact_outer_bottom_gap']
            for row in checkpoint_rows
        ),
        'certificate_route': 'componentwise_gs_increment_subeigenvector_tail_bound',
        'roadmap_status': 'secondary_tightening_gs_rate_closed_no_new_object',
    }

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_gs_rate_source_summary.csv'
    checkpoints_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_gs_rate_source_checkpoints.csv'

    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with checkpoints_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0].keys()))
        writer.writeheader()
        for row in checkpoint_rows:
            writer.writerow(row)

    print(summary_path)
    print(checkpoints_path)


if __name__ == '__main__':
    main()
