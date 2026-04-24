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
    strict_summary = next(csv.DictReader((OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_strict_slab_source_summary.csv').open()))

    D = float(strict_summary['D'])
    rho_anchor = float(strict_summary['rho_anchor'])
    zeta_anchor = float(strict_summary['zeta_anchor'])
    h_star = float(strict_summary['strict_cap'])
    req_global = float(strict_summary['required_global'])
    req_local = float(strict_summary['required_local'])
    barrier_mass = float(strict_summary['crude_torsion_barrier_mass'])

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
    Ksub = Klap[flat][:, flat].tocsr()
    Msub = m_mat[flat][:, flat].tocsr()
    phi_flat = phi_L.ravel()[flat]
    rhs_flat = rhs0.ravel()[flat]

    # Exact Dirichlet decomposition on the strict slab:
    #   -Delta_a u_sharp = f_sharp,  u_sharp|_∂=0
    #   -Delta_a h_sharp = 0,        h_sharp|_∂=phi_L|_∂
    source_potential = spsolve(Ksub, Msub @ rhs_flat)
    boundary_lift = phi_flat - source_potential
    reconstruction_resid = float(np.max(np.abs(phi_flat - (boundary_lift + source_potential))))

    # Equivalent boundary-defect formulation.
    boundary_defect = Ksub @ phi_flat - Msub @ rhs_flat
    boundary_lift_from_defect = spsolve(Ksub, boundary_defect)
    defect_route_resid = float(np.max(np.abs(boundary_lift - boundary_lift_from_defect)))

    weight_flat = (2.0 * w).ravel()[flat]
    I_strict = float(np.sum(weight_flat * phi_flat))
    I_boundary = float(np.sum(weight_flat * boundary_lift))
    I_source = float(np.sum(weight_flat * source_potential))

    remaining_deficit_global = req_global - I_boundary
    remaining_deficit_local = req_local - I_boundary

    row_rel = np.round((zz.ravel()[flat] - D / 2.0), 6)
    row_data = []
    for zv in sorted(np.unique(row_rel)):
        mask = row_rel == zv
        row_data.append({
            'z_rel': float(zv),
            'signed_mass_exact': float(np.sum(weight_flat[mask] * phi_flat[mask])),
            'signed_mass_boundary': float(np.sum(weight_flat[mask] * boundary_lift[mask])),
            'signed_mass_source': float(np.sum(weight_flat[mask] * source_potential[mask])),
            'min_phiL_row': float(np.min(phi_flat[mask])),
            'min_rhs0_row': float(np.min(rhs_flat[mask])),
        })

    summary = {
        'D': D,
        'rho_anchor': rho_anchor,
        'zeta_anchor': zeta_anchor,
        'strict_cap': h_star,
        'required_global': req_global,
        'required_local': req_local,
        'strict_slab_signed_mass': I_strict,
        'strict_slab_boundary_lift_mass': I_boundary,
        'strict_slab_source_potential_mass': I_source,
        'dirichlet_split_reconstruction_resid_abs': reconstruction_resid,
        'boundary_defect_nonnegative': int(np.all(boundary_defect >= -1e-12)),
        'boundary_defect_route_resid_abs': defect_route_resid,
        'boundary_lift_positive_everywhere': int(np.all(boundary_lift > 0.0)),
        'source_potential_positive_everywhere': int(np.all(source_potential > 0.0)),
        'min_boundary_lift': float(np.min(boundary_lift)),
        'min_source_potential': float(np.min(source_potential)),
        'boundary_lift_over_required_global': I_boundary / max(req_global, 1e-300),
        'boundary_lift_over_required_local': I_boundary / max(req_local, 1e-300),
        'source_potential_over_required_global': I_source / max(req_global, 1e-300),
        'source_potential_over_required_local': I_source / max(req_local, 1e-300),
        'boundary_lift_exact_mass_share': I_boundary / max(I_strict, 1e-300),
        'source_potential_exact_mass_share': I_source / max(I_strict, 1e-300),
        'global_deficit_after_boundary': remaining_deficit_global,
        'local_deficit_after_boundary': remaining_deficit_local,
        'source_over_global_deficit': I_source / max(remaining_deficit_global, 1e-300),
        'source_over_local_deficit': (I_source / remaining_deficit_local) if remaining_deficit_local > 0.0 else float('nan'),
        'boundary_alone_closes_global': int(I_boundary >= req_global),
        'boundary_alone_closes_local': int(I_boundary >= req_local),
        'crude_barrier_mass': barrier_mass,
        'boundary_plus_crude_barrier_over_required_global': (I_boundary + barrier_mass) / max(req_global, 1e-300),
        'preferred_parent_side_route': 'strict_slab_boundary_lift_plus_source_potential',
    }

    detail_rows = [
        {'term': 'strict_slab_signed_mass', 'value': I_strict, 'role': 'strict_slab_target'},
        {'term': 'strict_slab_boundary_lift_mass', 'value': I_boundary, 'role': 'Dirichlet_harmonic_lift'},
        {'term': 'strict_slab_source_potential_mass', 'value': I_source, 'role': 'zero_boundary_source_potential'},
        {'term': 'required_global', 'value': req_global, 'role': 'global_signed_mass_threshold'},
        {'term': 'required_local', 'value': req_local, 'role': 'local_signed_mass_threshold'},
        {'term': 'global_deficit_after_boundary', 'value': remaining_deficit_global, 'role': 'remaining_gap_after_boundary_only'},
        {'term': 'crude_barrier_mass', 'value': barrier_mass, 'role': 'naive_constant_source_barrier'},
    ]

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_dirichlet_split_source_summary.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_dirichlet_split_source_detail.csv'
    rows_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_strict_slab_dirichlet_split_source_rows.csv'

    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with detail_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)
    with rows_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_data[0].keys()))
        writer.writeheader()
        for row in row_data:
            writer.writerow(row)

    print(summary_path)
    print(detail_path)
    print(rows_path)


if __name__ == '__main__':
    main()
