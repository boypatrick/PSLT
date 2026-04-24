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
    boundary_lift = spsolve(Ksub, boundary_defect)

    weight_flat = (2.0 * w).ravel()[flat]
    I_boundary = float(np.sum(weight_flat * boundary_lift))

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

    category_rows = []
    split_sum = np.zeros_like(boundary_lift)
    for name, mask in categories.items():
        global_idx = np.flatnonzero(mask.ravel())
        local_idx = local_of_global[global_idx]
        local_idx = local_idx[local_idx >= 0]
        defect_part = np.zeros_like(boundary_defect)
        defect_part[local_idx] = boundary_defect[local_idx]
        harmonic_part = spsolve(Ksub, defect_part)
        split_sum += harmonic_part
        mass_part = float(np.sum(weight_flat * harmonic_part))
        category_rows.append(
            {
                'category': name,
                'boundary_point_count': int(local_idx.size),
                'harmonic_lift_mass': mass_part,
                'mass_share_of_boundary_lift': mass_part / max(I_boundary, 1e-300),
                'min_harmonic_part': float(np.min(harmonic_part)),
                'max_harmonic_part': float(np.max(harmonic_part)),
            }
        )

    split_resid = float(np.max(np.abs(split_sum - boundary_lift)))

    category_mass = {row['category']: row['harmonic_lift_mass'] for row in category_rows}
    I_top_cap = category_mass['top'] + category_mass['corner_lt'] + category_mass['corner_rt']
    I_physical = I_boundary - I_top_cap
    I_outer_bottom = (
        category_mass['outer_rho']
        + category_mass['bottom']
        + category_mass['corner_lb']
        + category_mass['corner_rb']
        + category_mass['axis']
    )

    summary = {
        'D': D,
        'strict_cap': h_star,
        'required_global': req_global,
        'required_local': req_local,
        'strict_slab_boundary_lift_mass': I_boundary,
        'strict_slab_source_potential_mass': I_source,
        'edge_split_reconstruction_resid_abs': split_resid,
        'boundary_mass_top_cap': I_top_cap,
        'boundary_mass_physical_edges': I_physical,
        'boundary_mass_outer_bottom_axis': I_outer_bottom,
        'physical_plus_source_over_required_global': (I_physical + I_source) / max(req_global, 1e-300),
        'physical_plus_source_over_required_local': (I_physical + I_source) / max(req_local, 1e-300),
        'physical_boundary_over_boundary_total': I_physical / max(I_boundary, 1e-300),
        'top_cap_over_boundary_total': I_top_cap / max(I_boundary, 1e-300),
        'outer_bottom_axis_over_boundary_total': I_outer_bottom / max(I_boundary, 1e-300),
        'top_cap_is_artificial_remainder': 1,
        'physical_plus_source_closes_global': int(I_physical + I_source >= req_global),
        'physical_plus_source_closes_local': int(I_physical + I_source >= req_local),
        'top_cap_mass_over_global_gap_after_physical_only': I_top_cap / max(req_global - I_physical, 1e-300),
        'outer_bottom_axis_plus_source_over_required_global': (I_outer_bottom + I_source) / max(req_global, 1e-300),
        'boundary_axis_mass': category_mass['axis'],
        'boundary_outer_mass': category_mass['outer_rho'],
        'boundary_bottom_mass': category_mass['bottom'],
        'boundary_top_mass': category_mass['top'],
        'boundary_corner_lb_mass': category_mass['corner_lb'],
        'boundary_corner_lt_mass': category_mass['corner_lt'],
        'boundary_corner_rb_mass': category_mass['corner_rb'],
        'boundary_corner_rt_mass': category_mass['corner_rt'],
    }

    detail_rows = category_rows + [
        {
            'category': 'physical_edges_group',
            'boundary_point_count': int(
                categories['axis'].sum()
                + categories['outer_rho'].sum()
                + categories['bottom'].sum()
                + categories['corner_lb'].sum()
                + categories['corner_rb'].sum()
            ),
            'harmonic_lift_mass': I_physical,
            'mass_share_of_boundary_lift': I_physical / max(I_boundary, 1e-300),
            'min_harmonic_part': float('nan'),
            'max_harmonic_part': float('nan'),
        },
        {
            'category': 'top_cap_group',
            'boundary_point_count': int(
                categories['top'].sum() + categories['corner_lt'].sum() + categories['corner_rt'].sum()
            ),
            'harmonic_lift_mass': I_top_cap,
            'mass_share_of_boundary_lift': I_top_cap / max(I_boundary, 1e-300),
            'min_harmonic_part': float('nan'),
            'max_harmonic_part': float('nan'),
        },
        {
            'category': 'source_potential',
            'boundary_point_count': 0,
            'harmonic_lift_mass': I_source,
            'mass_share_of_boundary_lift': I_source / max(I_boundary, 1e-300),
            'min_harmonic_part': float('nan'),
            'max_harmonic_part': float('nan'),
        },
    ]

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_boundary_edge_source_summary.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_boundary_edge_source_detail.csv'

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
