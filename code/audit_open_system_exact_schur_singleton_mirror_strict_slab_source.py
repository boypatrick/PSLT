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


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.zeros_like(mask, dtype=bool)
        if di == 1:
            shifted[:-1, :] = mask[1:, :]
        elif di == -1:
            shifted[1:, :] = mask[:-1, :]
        elif dj == 1:
            shifted[:, :-1] = mask[:, 1:]
        else:
            shifted[:, 1:] = mask[:, :-1]
        out |= mask & (~shifted)
    return out


def main() -> None:
    D = 11.0
    rho_anchor = 0.70
    zeta_anchor = 1.21

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
    lower = plus & (zz < D / 2.0)
    I_lower = float(np.sum(2.0 * w[lower] * phi_L[lower]))

    prior = next(csv.DictReader((OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_lobe_half_source_summary.csv').open()))
    req_global = float(prior['required_global'])
    req_local = float(prior['required_local'])

    E1 = float(evals[0])
    E2 = float(evals[1])
    Ebar = 0.5 * (E1 + E2)
    DeltaE = 0.5 * (E2 - E1)

    # Bare Laplacian route on the strict slab:
    # (-Delta_a) phi_L = (Ebar - U_a) phi_L - DeltaE * phi_R.
    rhs0 = (Ebar - uu) * phi_L - DeltaE * phi_R
    Klap = (k_mat - diags((rr * uu).ravel(), 0, format='csr')).tocsr()

    scan_rows = []
    best = None
    caps = [round(float(h), 2) for h in np.arange(level.dz, zeta_anchor + 1e-12, level.dz)]
    for h_cap in caps:
        slab = plus & ((zz - D / 2.0) <= -h_cap)
        if not np.any(slab):
            continue
        bdry = boundary_mask(slab)
        phiL_s = phi_L[slab]
        phiR_s = phi_R[slab]
        rhs_s = rhs0[slab]
        ratio_expr = (uu[slab] - Ebar) + DeltaE * (phi_R[slab] / np.maximum(phiL_s, 1e-300))
        kappa_needed = float(max(0.0, np.max(ratio_expr)))
        signed_mass = float(np.sum(2.0 * w[slab] * phiL_s))
        row = {
            'h_cap': float(h_cap),
            'signed_mass': signed_mass,
            'signed_mass_over_required_global': signed_mass / max(req_global, 1e-300),
            'signed_mass_over_required_local': signed_mass / max(req_local, 1e-300),
            'min_phiL': float(np.min(phiL_s)),
            'min_phiR': float(np.min(phiR_s)),
            'min_phiL_boundary': float(np.min(phi_L[bdry])),
            'min_phiR_boundary': float(np.min(phi_R[bdry])),
            'min_rhs0': float(np.min(rhs_s)),
            'max_rhs0': float(np.max(rhs_s)),
            'kappa_needed': kappa_needed,
            'all_phiL_positive': int(np.all(phiL_s > 0.0)),
            'all_phiR_positive': int(np.all(phiR_s > 0.0)),
            'boundary_phiL_positive': int(np.all(phi_L[bdry] > 0.0)),
            'boundary_phiR_positive': int(np.all(phi_R[bdry] > 0.0)),
            'rhs0_nonnegative': int(np.all(rhs_s >= -1e-14)),
            'admissible_global': int(
                np.all(phiL_s > 0.0)
                and np.all(phiR_s > 0.0)
                and np.all(phi_L[bdry] > 0.0)
                and np.all(phi_R[bdry] > 0.0)
                and np.all(rhs_s >= -1e-14)
                and signed_mass >= req_global
            ),
            'point_count': int(np.count_nonzero(slab)),
            'boundary_count': int(np.count_nonzero(bdry)),
        }
        scan_rows.append(row)
        if best is None and row['admissible_global'] == 1:
            best = row

    if best is None:
        raise RuntimeError('No admissible strict slab found for the bare Laplacian route.')

    h_star = float(best['h_cap'])
    strict = plus & ((zz - D / 2.0) <= -h_star)
    strict_bdry = boundary_mask(strict)
    phiL_strict = phi_L[strict]
    phiR_strict = phi_R[strict]
    rhs_strict = rhs0[strict]
    I_strict = float(np.sum(2.0 * w[strict] * phiL_strict))
    strip_mass = I_lower - I_strict

    flat = np.flatnonzero(strict.ravel())
    Ksub = Klap[flat][:, flat].tocsr()
    Msub = m_mat[flat][:, flat].tocsr()
    tau = spsolve(Ksub, Msub @ np.ones(len(flat)))
    g_min = float(np.min(rhs_strict))
    barrier_field = g_min * tau
    phiL_flat = phi_L.ravel()[flat]
    barrier_comp = phiL_flat - barrier_field
    I_barrier = float(np.sum(2.0 * w.ravel()[flat] * barrier_field))

    summary = {
        'D': D,
        'rho_anchor': rho_anchor,
        'zeta_anchor': zeta_anchor,
        'strict_cap': h_star,
        'E1': E1,
        'E2': E2,
        'Ebar': Ebar,
        'DeltaE': DeltaE,
        'required_global': req_global,
        'required_local': req_local,
        'lower_half_signed_mass': I_lower,
        'strict_slab_signed_mass': I_strict,
        'strict_slab_over_required_global': I_strict / max(req_global, 1e-300),
        'strict_slab_over_required_local': I_strict / max(req_local, 1e-300),
        'omitted_interface_strip_mass': strip_mass,
        'omitted_strip_over_lower_half': strip_mass / max(I_lower, 1e-300),
        'omitted_strip_over_required_global': strip_mass / max(req_global, 1e-300),
        'phiL_positive_on_strict_slab': int(np.all(phiL_strict > 0.0)),
        'phiR_positive_on_strict_slab': int(np.all(phiR_strict > 0.0)),
        'phiL_positive_on_strict_boundary': int(np.all(phi_L[strict_bdry] > 0.0)),
        'phiR_positive_on_strict_boundary': int(np.all(phi_R[strict_bdry] > 0.0)),
        'min_phiL_strict_slab': float(np.min(phiL_strict)),
        'min_phiR_strict_slab': float(np.min(phiR_strict)),
        'min_phiL_strict_boundary': float(np.min(phi_L[strict_bdry])),
        'min_phiR_strict_boundary': float(np.min(phi_R[strict_bdry])),
        'min_rhs0_strict_slab': float(np.min(rhs_strict)),
        'max_rhs0_strict_slab': float(np.max(rhs_strict)),
        'kappa_needed_strict': float(max(0.0, np.max((uu[strict] - Ebar) + DeltaE * (phiR_strict / np.maximum(phiL_strict, 1e-300))))),
        'strict_slab_point_count': int(np.count_nonzero(strict)),
        'strict_slab_boundary_count': int(np.count_nonzero(strict_bdry)),
        'crude_torsion_barrier_mass': I_barrier,
        'crude_torsion_barrier_over_strict_mass': I_barrier / max(I_strict, 1e-300),
        'crude_torsion_barrier_over_required_global': I_barrier / max(req_global, 1e-300),
        'crude_torsion_barrier_comp_min': float(np.min(barrier_comp)),
        'preferred_parent_side_route': 'strict_lower_slab_laplacian_barrier',
    }

    detail_rows = [
        {'term': 'lower_half_signed_mass', 'value': I_lower, 'role': 'previous_parent_target'},
        {'term': 'strict_slab_signed_mass', 'value': I_strict, 'role': 'strict_barrier_domain_target'},
        {'term': 'omitted_interface_strip_mass', 'value': strip_mass, 'role': 'interface_strip_remainder'},
        {'term': 'required_global', 'value': req_global, 'role': 'global_signed_mass_threshold'},
        {'term': 'required_local', 'value': req_local, 'role': 'local_signed_mass_threshold'},
        {'term': 'min_rhs0_strict_slab', 'value': float(np.min(rhs_strict)), 'role': 'bare_laplacian_source_floor'},
        {'term': 'crude_torsion_barrier_mass', 'value': I_barrier, 'role': 'constant_source_barrier_lower_bound'},
    ]

    summary_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_strict_slab_source_summary.csv'
    detail_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_strict_slab_source_detail.csv'
    scan_path = OUT_DIR / 'chi_open_system_exact_schur_singleton_mirror_strict_slab_source_scan.csv'

    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with detail_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)
    with scan_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scan_rows[0].keys()))
        writer.writeheader()
        for row in scan_rows:
            writer.writerow(row)

    print(summary_path)
    print(detail_path)
    print(scan_path)


if __name__ == '__main__':
    main()
