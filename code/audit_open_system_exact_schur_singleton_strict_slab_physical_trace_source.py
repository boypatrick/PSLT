#!/usr/bin/env python3
from __future__ import annotations

import csv
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


def main() -> None:
    strict_summary = next(
        csv.DictReader(
            (OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_strict_slab_source_summary.csv').open()
        )
    )
    dirichlet_summary = next(
        csv.DictReader(
            (OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_dirichlet_split_source_summary.csv').open()
        )
    )
    edge_summary = next(
        csv.DictReader(
            (OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_boundary_edge_source_summary.csv').open()
        )
    )

    D = float(strict_summary['D'])
    rho_anchor = float(strict_summary['rho_anchor'])
    zeta_anchor = float(strict_summary['zeta_anchor'])
    h_star = float(strict_summary['strict_cap'])
    req_global = float(strict_summary['required_global'])
    req_local = float(strict_summary['required_local'])
    I_source = float(dirichlet_summary['strict_slab_source_potential_mass'])

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
    psi1 = psi[:, :, 0]
    psi2 = psi[:, :, 1]
    phi_L = (psi1 - psi2) / np.sqrt(2.0)
    phi_R = (psi1 + psi2) / np.sqrt(2.0)

    w = 2.0 * np.pi * rr * level.dr * level.dz
    plus = (rr <= rho_anchor) & (np.abs(zz - D / 2.0) <= zeta_anchor)
    strict = plus & ((zz - D / 2.0) <= -h_star)

    E1 = float(evals[0])
    E2 = float(evals[1])
    Ebar = 0.5 * (E1 + E2)
    DeltaE = 0.5 * (E2 - E1)

    rhs0 = (Ebar - uu) * phi_L - DeltaE * phi_R
    Klap = (k_mat - diags((rr * uu).ravel(), 0, format='csr')).tocsr()

    flat = np.flatnonzero(strict.ravel())
    local_of_global = -np.ones(strict.size, dtype=int)
    local_of_global[flat] = np.arange(len(flat))

    Ksub = Klap[flat][:, flat].tocsr()
    Msub = m_mat[flat][:, flat].tocsr()
    phi_flat = phi_L.ravel()[flat]
    rhs_flat = rhs0.ravel()[flat]
    boundary_defect = Ksub @ phi_flat - Msub @ rhs_flat
    weight_flat = (2.0 * w).ravel()[flat]

    # Adjoint Poisson weights: mass(h)=p^T K h, so edge traces are p^T b_edge.
    poisson_weight = spsolve(Ksub.T, weight_flat)
    boundary_lift = spsolve(Ksub, boundary_defect)
    I_boundary = float(np.sum(weight_flat * boundary_lift))
    I_boundary_adjoint = float(poisson_weight @ boundary_defect)

    up_in = np.zeros_like(strict, dtype=bool)
    up_in[:, :-1] = strict[:, 1:]
    down_in = np.zeros_like(strict, dtype=bool)
    down_in[:, 1:] = strict[:, :-1]
    right_in = np.zeros_like(strict, dtype=bool)
    right_in[:-1, :] = strict[1:, :]

    ii, jj = np.indices(strict.shape)
    boundary = strict & (~up_in | ~down_in | ~right_in | (ii == 0))
    on_axis = strict & (ii == 0)
    on_top = strict & ~up_in
    on_bottom = strict & ~down_in
    on_outer = strict & ~right_in

    categories = {
        'corner_lb': boundary & on_axis & on_bottom,
        'corner_lt': boundary & on_axis & on_top,
        'corner_rb': boundary & on_outer & on_bottom,
        'corner_rt': boundary & on_outer & on_top,
        'axis': boundary & on_axis & ~on_top & ~on_bottom,
        'outer_rho': boundary & on_outer & ~on_top & ~on_bottom,
        'bottom': boundary & on_bottom & ~on_axis & ~on_outer,
        'top': boundary & on_top & ~on_axis & ~on_outer,
    }
    physical_names = ['axis', 'outer_rho', 'bottom', 'corner_lb', 'corner_rb']

    detail_rows = []
    physical_local_idx: list[int] = []
    for name, mask in categories.items():
        global_idx = np.flatnonzero(mask.ravel())
        local_idx = local_of_global[global_idx]
        local_idx = local_idx[local_idx >= 0]
        if name in physical_names:
            physical_local_idx.extend(local_idx.tolist())
        defect_sum = float(np.sum(boundary_defect[local_idx]))
        exact_mass = float(poisson_weight[local_idx] @ boundary_defect[local_idx])
        avg_poisson_weight = exact_mass / max(defect_sum, 1e-300)
        min_poisson_weight = float(np.min(poisson_weight[local_idx])) if local_idx.size else float('nan')
        min_product_mass = min_poisson_weight * defect_sum if local_idx.size else float('nan')
        detail_rows.append(
            {
                'category': name,
                'boundary_point_count': int(local_idx.size),
                'boundary_defect_sum': defect_sum,
                'poisson_weighted_mass': exact_mass,
                'average_poisson_weight': avg_poisson_weight,
                'min_poisson_weight': min_poisson_weight,
                'min_weight_product_mass': min_product_mass,
                'min_boundary_defect': float(np.min(boundary_defect[local_idx])) if local_idx.size else float('nan'),
                'max_boundary_defect': float(np.max(boundary_defect[local_idx])) if local_idx.size else float('nan'),
            }
        )

    physical_local = np.array(physical_local_idx, dtype=int)
    I_phys = float(poisson_weight[physical_local] @ boundary_defect[physical_local])
    B_phys = float(np.sum(boundary_defect[physical_local]))
    avg_phys = I_phys / max(B_phys, 1e-300)

    edge_mass = {row['category']: row['poisson_weighted_mass'] for row in detail_rows}
    I_outer_bottom = edge_mass['outer_rho'] + edge_mass['bottom']
    I_lower_corners = edge_mass['corner_lb'] + edge_mass['corner_rb']
    I_outer_bottom_rb = I_outer_bottom + edge_mass['corner_rb']
    physical_target_global_with_source = req_global - I_source
    physical_target_local_with_source = req_local - I_source

    summary = {
        'D': D,
        'strict_cap': h_star,
        'required_global': req_global,
        'required_local': req_local,
        'source_potential_mass': I_source,
        'physical_trace_required_global_after_source': physical_target_global_with_source,
        'physical_trace_required_local_after_source': physical_target_local_with_source,
        'boundary_lift_mass': I_boundary,
        'boundary_lift_adjoint_mass': I_boundary_adjoint,
        'adjoint_representation_resid_abs': abs(I_boundary - I_boundary_adjoint),
        'physical_trace_mass': I_phys,
        'physical_trace_mass_from_edge_summary': float(edge_summary['boundary_mass_physical_edges']),
        'physical_trace_resid_vs_edge_summary_abs': abs(I_phys - float(edge_summary['boundary_mass_physical_edges'])),
        'physical_trace_over_required_global_after_source': I_phys / max(physical_target_global_with_source, 1e-300),
        'physical_trace_over_required_local_after_source': I_phys / max(physical_target_local_with_source, 1e-300),
        'physical_plus_source_over_required_global': (I_phys + I_source) / max(req_global, 1e-300),
        'physical_plus_source_over_required_local': (I_phys + I_source) / max(req_local, 1e-300),
        'physical_boundary_defect_sum': B_phys,
        'physical_average_poisson_weight': avg_phys,
        'outer_wall_mass': edge_mass['outer_rho'],
        'bottom_edge_mass': edge_mass['bottom'],
        'lower_corner_mass': I_lower_corners,
        'axis_mass': edge_mass['axis'],
        'outer_bottom_mass': I_outer_bottom,
        'outer_bottom_over_required_global_after_source': I_outer_bottom / max(physical_target_global_with_source, 1e-300),
        'outer_bottom_right_corner_mass': I_outer_bottom_rb,
        'outer_bottom_right_corner_over_required_global_after_source': I_outer_bottom_rb / max(
            physical_target_global_with_source, 1e-300
        ),
        'remaining_gap_after_outer_bottom_right_corner': physical_target_global_with_source - I_outer_bottom_rb,
        'lower_left_corner_mass': edge_mass['corner_lb'],
        'outer_wall_share_of_physical': edge_mass['outer_rho'] / max(I_phys, 1e-300),
        'bottom_edge_share_of_physical': edge_mass['bottom'] / max(I_phys, 1e-300),
        'lower_corner_share_of_physical': I_lower_corners / max(I_phys, 1e-300),
        'preferred_parent_side_route': 'physical_boundary_adjoint_poisson_trace_plus_source',
    }

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_physical_trace_source_summary.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_physical_trace_source_detail.csv'

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
