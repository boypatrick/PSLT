#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

from extract_chi_open_system_micro import (
    Level,
    PhysicalParams,
    build_generalized_operator,
    normalize_modes,
    spherical_average_v_eff,
)

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'chi_open_system'


def load_fields(D: float):
    p = PhysicalParams()
    level = Level('fine', dr=0.06, dz=0.03)
    rho_max = 3.0
    z_margin = 6.0
    sigma = 2.5
    z_max = D / 2.0 + z_margin
    rho, z, rr, zz, uu, k_mat, m_mat = build_generalized_operator(
        D=D, p=p, rho_max=rho_max, z_max=z_max, dr=level.dr, dz=level.dz
    )
    evals, evecs = eigsh(k_mat, k=2, M=m_mat, sigma=sigma, which='LM', tol=1e-8, maxiter=30000)
    idx = np.argsort(np.real(evals))
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    psi = np.stack([evecs[:, 0].reshape(len(rho), len(z)), evecs[:, 1].reshape(len(rho), len(z))], axis=2)
    psi = normalize_modes(psi, rho, level.dr, level.dz)
    psi1 = psi[:, :, 0]
    psi2 = psi[:, :, 1]
    phi_R = (psi1 + psi2) / math.sqrt(2.0)
    phi_L = (psi1 - psi2) / math.sqrt(2.0)
    w2 = 2.0 * math.pi * rho[:, None] * level.dr * level.dz * np.ones_like(zz)

    vfull = uu + p.m0 * p.m0
    r_field = np.sqrt(rr * rr + zz * zz)
    r_line = np.linspace(0.0, float(r_field.max()), 1800)
    vbar_line = spherical_average_v_eff(r_line, D, p, n_mu=120)
    vbar = np.interp(r_field.ravel(), r_line, vbar_line).reshape(r_field.shape)
    delta_v = vfull - vbar

    contrast = psi1**2 - psi2**2
    overlap_density = 2.0 * phi_R * phi_L
    integrand = w2 * delta_v * contrast
    exact = float(np.sum(integrand))

    return {
        'D': D,
        'rho': rho,
        'z': z,
        'rr': rr,
        'zz': zz,
        'psi1': psi1,
        'psi2': psi2,
        'phi_R': phi_R,
        'phi_L': phi_L,
        'w2': w2,
        'delta_v': delta_v,
        'contrast': contrast,
        'overlap_density': overlap_density,
        'integrand': integrand,
        'exact': exact,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ds', default='5,11,18')
    args = ap.parse_args()
    Ds = [float(x.strip()) for x in args.Ds.split(',') if x.strip()]

    detail_rows = []
    summary_rows = []
    for D in Ds:
        data = load_fields(D)
        rr = data['rr']
        zz = data['zz']
        z = data['z']
        integrand = data['integrand']
        exact = data['exact']
        abs_total = float(np.sum(np.abs(integrand)))
        overlap_resid = float(np.max(np.abs(data['contrast'] - data['overlap_density'])))
        zsum = np.sum(integrand, axis=0)

        for family, params in [('zslab', [0.5,1.0,1.5,2.0]), ('core_ball', [0.5,0.75,1.0,1.25,1.5,2.0])]:
            for R in params:
                if family == 'zslab':
                    right = np.abs(zz - D/2.0) <= R
                    left = np.abs(zz + D/2.0) <= R
                    mid = np.abs(zz) <= R
                    union = right | left
                else:
                    rp = np.sqrt(rr * rr + (zz - D/2.0) ** 2)
                    rm = np.sqrt(rr * rr + (zz + D/2.0) ** 2)
                    right = rp <= R
                    left = rm <= R
                    mid = np.abs(zz) <= R
                    union = right | left
                tail = ~(union if family == 'core_ball' else (right | left | mid))
                right_signed = float(np.sum(integrand[right]))
                left_signed = float(np.sum(integrand[left]))
                mid_signed = float(np.sum(integrand[mid]))
                if family == 'zslab':
                    outer_signed = float(np.sum(integrand[tail]))
                    outer_abs = float(np.sum(np.abs(integrand[tail])))
                else:
                    outer_signed = float(np.sum(integrand[tail]))
                    outer_abs = float(np.sum(np.abs(integrand[tail])))
                union_abs = float(np.sum(np.abs(integrand[union])))
                mid_abs = float(np.sum(np.abs(integrand[mid])))
                detail_rows.append({
                    'D': D,
                    'family': family,
                    'R': R,
                    'exact': exact,
                    'max_overlap_identity_residual_abs': overlap_resid,
                    'right_signed': right_signed,
                    'left_signed': left_signed,
                    'mid_signed': mid_signed,
                    'outer_signed': outer_signed,
                    'union_abs_share': union_abs / max(abs_total, 1e-300),
                    'mid_abs_share': mid_abs / max(abs_total, 1e-300),
                    'outer_abs_share': outer_abs / max(abs_total, 1e-300),
                })
        # quadratic jet on right core ball
        for R in [0.75,1.0,1.25,1.5]:
            zeta = zz - D/2.0
            core = (rr**2 + zeta**2) <= R**2
            X = np.column_stack([np.ones(core.sum()), rr[core]**2, zeta[core]**2])
            y = data['delta_v'][core].ravel()
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a0, ar, az = coef
            jet = a0 + ar * rr**2 + az * zeta**2
            right_exact = float(np.sum(integrand[core]))
            right_jet = float(np.sum(data['w2'][core] * jet[core] * data['contrast'][core]))
            union_exact = 2.0 * right_exact
            union_jet = 2.0 * right_jet
            tail_signed = exact - union_exact
            detail_rows.append({
                'D': D,
                'family': 'quad_jet_ball',
                'R': R,
                'exact': exact,
                'max_overlap_identity_residual_abs': overlap_resid,
                'right_signed': right_exact,
                'left_signed': right_exact,
                'mid_signed': np.nan,
                'outer_signed': tail_signed,
                'union_abs_share': np.nan,
                'mid_abs_share': np.nan,
                'outer_abs_share': np.nan,
                'union_exact': union_exact,
                'union_jet': union_jet,
                'rel_jet_err_vs_union': abs(union_jet - union_exact) / max(abs(union_exact), 1e-300),
                'rel_jet_err_vs_exact': abs(union_jet - exact) / max(abs(exact), 1e-300),
                'a0': float(a0),
                'ar': float(ar),
                'az': float(az),
            })
        # z-profile peaks
        top_idx = np.argsort(np.abs(zsum))[::-1][:12]
        for rank, j in enumerate(top_idx, start=1):
            detail_rows.append({
                'D': D,
                'family': 'z_profile_peak',
                'R': float(rank),
                'exact': exact,
                'max_overlap_identity_residual_abs': overlap_resid,
                'z_peak': float(z[j]),
                'I_z_peak': float(zsum[j]),
            })
        summary_rows.append({
            'D': D,
            'exact': exact,
            'max_overlap_identity_residual_abs': overlap_resid,
        })

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / 'chi_open_system_parity_contrast_core_jet_probe_detail.csv'
    summary_path = OUTDIR / 'chi_open_system_parity_contrast_core_jet_probe_summary.csv'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == '__main__':
    main()
