#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit_open_system_parity_contrast_self_mirror_source import solve_fields
from extract_chi_localized_2d import Level, PhysicalParams

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'chi_open_system'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ds', default='4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20')
    ap.add_argument('--rho-anchor', type=float, default=0.70)
    ap.add_argument('--zeta-cut', type=float, default=1.20)
    ap.add_argument('--rho-band', type=float, default=0.20)
    ap.add_argument('--zeta-band-low', type=float, default=0.10)
    ap.add_argument('--rho-max-grid', type=float, default=3.0)
    ap.add_argument('--z-margin', type=float, default=6.0)
    ap.add_argument('--dr', type=float, default=0.06)
    ap.add_argument('--dz', type=float, default=0.03)
    ap.add_argument('--sigma', type=float, default=2.5)
    ap.add_argument('--tol', type=float, default=1e-8)
    ap.add_argument('--maxiter', type=int, default=30000)
    args = ap.parse_args()

    Ds = [float(x.strip()) for x in args.Ds.split(',') if x.strip()]
    p = PhysicalParams()
    level = Level('fine', dr=float(args.dr), dz=float(args.dz))

    rows = []
    for D in Ds:
        f = solve_fields(
            D=D,
            p=p,
            level=level,
            rho_max=float(args.rho_max_grid),
            z_margin=float(args.z_margin),
            sigma=float(args.sigma),
            tol=float(args.tol),
            maxiter=int(args.maxiter),
        )
        rr = f['rr']
        zz = f['zz']
        psi1 = f['psi1']
        psi2 = f['psi2']

        plus = (rr <= float(args.rho_anchor)) & (np.abs(zz - D / 2.0) <= float(args.zeta_cut))
        minus = (rr <= float(args.rho_anchor)) & (np.abs(zz + D / 2.0) <= float(args.zeta_cut))
        inner = plus | minus
        zeta_p = np.abs(zz - D / 2.0)
        zeta_m = np.abs(zz + D / 2.0)
        local_zeta = np.where(plus, zeta_p, zeta_m)
        band = inner & (rr >= float(args.rho_band)) & (local_zeta >= float(args.zeta_band_low)) & (local_zeta <= float(args.zeta_cut))

        phiA = (psi1 + psi2) / math.sqrt(2.0)
        phiB = (psi1 - psi2) / math.sqrt(2.0)
        corep = (rr <= 0.35) & (np.abs(zz - D / 2.0) <= 0.35)
        corem = (rr <= 0.35) & (np.abs(zz + D / 2.0) <= 0.35)
        A_right = float(np.sum(phiA[corep] ** 2))
        A_left = float(np.sum(phiA[corem] ** 2))
        if A_right >= A_left:
            phiR, phiL = phiA, phiB
        else:
            phiR, phiL = phiB, phiA

        self_amp = np.zeros_like(rr)
        mir_amp = np.zeros_like(rr)
        self_amp[plus] = np.abs(phiR[plus])
        self_amp[minus] = np.abs(phiL[minus])
        mir_amp[plus] = np.abs(phiL[plus])
        mir_amp[minus] = np.abs(phiR[minus])

        M_abs_density = 2.0 * np.abs(phiR * phiL) * level.dr * level.dz
        factor_density = 2.0 * self_amp * mir_amp * level.dr * level.dz
        M_band_abs = float(np.sum(M_abs_density[band]))
        factorized_abs = float(np.sum(factor_density[band]))
        S_self = float(np.sum((self_amp * level.dr * level.dz)[band]))
        S_mir = float(np.sum((mir_amp * level.dr * level.dz)[band]))
        mir_avg_w_self = M_band_abs / max(2.0 * S_self, 1e-300)
        self_avg_w_mir = M_band_abs / max(2.0 * S_mir, 1e-300)

        rows.append(
            {
                'D': D,
                'M_band_abs': M_band_abs,
                'factorized_abs': factorized_abs,
                'factor_resid_abs': abs(M_band_abs - factorized_abs),
                'S_self': S_self,
                'S_mir': S_mir,
                'mir_avg_w_self': mir_avg_w_self,
                'self_avg_w_mir': self_avg_w_mir,
                'product_self_miravg': 2.0 * S_self * mir_avg_w_self,
                'product_mir_selfavg': 2.0 * S_mir * self_avg_w_mir,
            }
        )

    detail = pd.DataFrame(rows).sort_values('D').reset_index(drop=True)
    abs_floor_idx = int(detail['M_band_abs'].idxmin())
    self_floor_idx = int(detail['S_self'].idxmin())
    mir_floor_idx = int(detail['S_mir'].idxmin())
    mir_avg_floor_idx = int(detail['mir_avg_w_self'].idxmin())
    self_avg_floor_idx = int(detail['self_avg_w_mir'].idxmin())

    induced_self_route = 2.0 * float(detail['S_self'].min()) * float(detail['mir_avg_w_self'].min())
    induced_mir_route = 2.0 * float(detail['S_mir'].min()) * float(detail['self_avg_w_mir'].min())

    summary = pd.DataFrame(
        [
            {
                'max_factor_resid_abs': float(detail['factor_resid_abs'].max()),
                'M_band_abs_floor_abs': float(detail['M_band_abs'].min()),
                'M_band_abs_floor_D': float(detail.loc[abs_floor_idx, 'D']),
                'S_self_floor_abs': float(detail['S_self'].min()),
                'S_self_floor_D': float(detail.loc[self_floor_idx, 'D']),
                'S_mir_floor_abs': float(detail['S_mir'].min()),
                'S_mir_floor_D': float(detail.loc[mir_floor_idx, 'D']),
                'mir_avg_w_self_min': float(detail['mir_avg_w_self'].min()),
                'mir_avg_w_self_min_D': float(detail.loc[mir_avg_floor_idx, 'D']),
                'mir_avg_w_self_max': float(detail['mir_avg_w_self'].max()),
                'self_avg_w_mir_min': float(detail['self_avg_w_mir'].min()),
                'self_avg_w_mir_min_D': float(detail.loc[self_avg_floor_idx, 'D']),
                'self_avg_w_mir_max': float(detail['self_avg_w_mir'].max()),
                'induced_self_route_abs': induced_self_route,
                'exact_over_induced_self_route': float(detail['M_band_abs'].min()) / max(induced_self_route, 1e-300),
                'induced_mir_route_abs': induced_mir_route,
                'exact_over_induced_mir_route': float(detail['M_band_abs'].min()) / max(induced_mir_route, 1e-300),
            }
        ]
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_absolute_overlap_source_detail.csv'
    summary_path = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_absolute_overlap_source_summary.csv'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == '__main__':
    main()
