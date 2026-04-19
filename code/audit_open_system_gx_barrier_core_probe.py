#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

from extract_chi_open_system_micro import Level, PhysicalParams, build_generalized_operator, normalize_modes, spherical_average_v_eff

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'chi_open_system'


def compute_row(D: float) -> dict[str, float]:
    p = PhysicalParams()
    level = Level('fine', dr=0.06, dz=0.03)
    rho_max = 3.0
    z_margin = 6.0
    n_mu = 120
    tol = 1e-8
    maxiter = 30000
    sigma = 2.5

    z_max = D / 2.0 + z_margin
    rho, z, rr, zz, uu, k_mat, m_mat = build_generalized_operator(
        D=D, p=p, rho_max=rho_max, z_max=z_max, dr=level.dr, dz=level.dz
    )
    evals, evecs = eigsh(k_mat, k=2, M=m_mat, sigma=sigma, which='LM', tol=tol, maxiter=maxiter)
    idx = np.argsort(np.real(evals))
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    nr, nz = len(rho), len(z)
    psi = np.stack([evecs[:, 0].reshape(nr, nz), evecs[:, 1].reshape(nr, nz)], axis=2)
    psi = normalize_modes(psi, rho, level.dr, level.dz)
    psi1 = psi[:, :, 0]
    psi2 = psi[:, :, 1]

    w = 2.0 * math.pi * rho[:, None] * level.dr * level.dz
    vfull = uu + p.m0 * p.m0
    r_field = np.sqrt(rr * rr + zz * zz)
    r_line = np.linspace(0.0, float(r_field.max()), 1800)
    vbar_line = spherical_average_v_eff(r_line, D, p, n_mu=n_mu)
    vbar = np.interp(r_field.ravel(), r_line, vbar_line).reshape(r_field.shape)
    delta_v = vfull - vbar

    # Exact localized-overlap identity suggested by the user.
    phi_R = (psi1 + psi2) / math.sqrt(2.0)
    phi_L = (psi1 - psi2) / math.sqrt(2.0)
    contrast = psi1**2 - psi2**2
    overlap_density = 2.0 * phi_R * phi_L
    integrand = delta_v * contrast

    exact = float(np.sum(w * integrand))
    overlap_identity_residual = float(np.max(np.abs(contrast - overlap_density)))

    slab = np.abs(zz) <= D / 4.0
    exact_abs = abs(exact)
    abs_total = float(np.sum(w * np.abs(integrand)))
    slab_signed = float(np.sum(w * integrand * slab))
    slab_abs = float(np.sum(w * np.abs(integrand) * slab))
    tail_abs = float(np.sum(w * np.abs(integrand) * (~slab)))

    # Conservative barrier-slab lower estimate with the sign aligned to the exact integral.
    sign = 1.0 if exact >= 0 else -1.0
    J = sign * integrand
    slab_good = slab & (J > 0.0)
    slab_bad = slab & (J < 0.0)
    if np.any(slab_good):
        delta_min_good = float(np.min(np.abs(delta_v[slab_good])))
    else:
        delta_min_good = 0.0
    lower_good = float(delta_min_good * np.sum(w * np.abs(contrast) * slab_good))
    penalty = float(np.sum(w * np.abs(integrand) * (slab_bad | (~slab))))
    lower_bound = max(lower_good - penalty, 0.0)

    return {
        'D': float(D),
        'exact_V11_minus_V22': exact,
        'max_overlap_identity_residual_abs': overlap_identity_residual,
        'barrier_signed_share': abs(slab_signed) / max(exact_abs, 1e-300),
        'barrier_abs_share': slab_abs / max(abs_total, 1e-300),
        'tail_abs_share': tail_abs / max(abs_total, 1e-300),
        'barrier_core_lower_bound': lower_bound,
        'barrier_core_lower_over_exactabs': lower_bound / max(exact_abs, 1e-300),
        'barrier_good_deltaV_min_abs': delta_min_good,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ds', default='5,11,18')
    args = ap.parse_args()
    Ds = [float(x.strip()) for x in args.Ds.split(',') if x.strip()]
    rows = [compute_row(D) for D in Ds]
    df = pd.DataFrame(rows)
    detail = OUTDIR / 'chi_open_system_gx_barrier_core_probe_detail.csv'
    summary = OUTDIR / 'chi_open_system_gx_barrier_core_probe_summary.csv'
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(detail, index=False)
    pd.DataFrame([{
        'max_overlap_identity_residual_abs': float(df['max_overlap_identity_residual_abs'].max()),
        'min_barrier_abs_share': float(df['barrier_abs_share'].min()),
        'max_barrier_abs_share': float(df['barrier_abs_share'].max()),
        'min_tail_abs_share': float(df['tail_abs_share'].min()),
        'max_tail_abs_share': float(df['tail_abs_share'].max()),
        'all_barrier_core_lower_bounds_zero': int(np.allclose(df['barrier_core_lower_bound'], 0.0)),
    }]).to_csv(summary, index=False)
    print(detail)
    print(summary)


if __name__ == '__main__':
    main()
