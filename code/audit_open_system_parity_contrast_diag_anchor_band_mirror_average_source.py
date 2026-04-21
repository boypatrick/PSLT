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
        if float(np.sum(phiA[corep] ** 2)) >= float(np.sum(phiA[corem] ** 2)):
            phiR, phiL = phiA, phiB
        else:
            phiR, phiL = phiB, phiA

        self_amp = np.zeros_like(rr)
        mir_amp = np.zeros_like(rr)
        self_amp[plus] = np.abs(phiR[plus])
        self_amp[minus] = np.abs(phiL[minus])
        mir_amp[plus] = np.abs(phiL[plus])
        mir_amp[minus] = np.abs(phiR[minus])

        area_band = float(np.sum(band) * level.dr * level.dz)
        S_self = float(np.sum(self_amp[band]) * level.dr * level.dz)
        S_mir = float(np.sum(mir_amp[band]) * level.dr * level.dz)
        M_abs = float(np.sum((2.0 * self_amp * mir_amp)[band]) * level.dr * level.dz)

        mirror_avg = S_mir / max(area_band, 1e-300)
        mirror_avg_weighted = M_abs / max(2.0 * S_self, 1e-300)
        self_bias = mirror_avg_weighted / max(mirror_avg, 1e-300)
        factor_resid = abs(mirror_avg_weighted - self_bias * mirror_avg)

        rows.append(
            {
                'D': D,
                'A_band': area_band,
                'S_self': S_self,
                'S_mir': S_mir,
                'M_band_abs': M_abs,
                'mirror_avg': mirror_avg,
                'mirror_avg_weighted': mirror_avg_weighted,
                'self_bias': self_bias,
                'factor_resid_abs': factor_resid,
            }
        )

    detail = pd.DataFrame(rows).sort_values('D').reset_index(drop=True)
    weighted_floor_idx = int(detail['mirror_avg_weighted'].idxmin())
    mean_floor_idx = int(detail['mirror_avg'].idxmin())
    bias_floor_idx = int(detail['self_bias'].idxmin())
    bias_ceil_idx = int(detail['self_bias'].idxmax())
    area_min_idx = int(detail['A_band'].idxmin())
    area_max_idx = int(detail['A_band'].idxmax())

    induced_lower = float(detail['mirror_avg'].min()) * float(detail['self_bias'].min())
    induced_from_Smir_area = float(detail['S_mir'].min()) / max(float(detail['A_band'].max()), 1e-300) * float(detail['self_bias'].min())

    summary = pd.DataFrame(
        [
            {
                'max_factor_resid_abs': float(detail['factor_resid_abs'].max()),
                'mirror_avg_weighted_floor_abs': float(detail['mirror_avg_weighted'].min()),
                'mirror_avg_weighted_floor_D': float(detail.loc[weighted_floor_idx, 'D']),
                'mirror_avg_floor_abs': float(detail['mirror_avg'].min()),
                'mirror_avg_floor_D': float(detail.loc[mean_floor_idx, 'D']),
                'self_bias_min': float(detail['self_bias'].min()),
                'self_bias_min_D': float(detail.loc[bias_floor_idx, 'D']),
                'self_bias_max': float(detail['self_bias'].max()),
                'self_bias_max_D': float(detail.loc[bias_ceil_idx, 'D']),
                'A_band_min': float(detail['A_band'].min()),
                'A_band_min_D': float(detail.loc[area_min_idx, 'D']),
                'A_band_max': float(detail['A_band'].max()),
                'A_band_max_D': float(detail.loc[area_max_idx, 'D']),
                'A_band_relative_spread': (float(detail['A_band'].max()) - float(detail['A_band'].min())) / max(float(detail['A_band'].mean()), 1e-300),
                'induced_lower_bound_abs': induced_lower,
                'exact_over_induced_lower_bound': float(detail['mirror_avg_weighted'].min()) / max(induced_lower, 1e-300),
                'induced_from_Smir_area_abs': induced_from_Smir_area,
                'exact_over_induced_from_Smir_area': float(detail['mirror_avg_weighted'].min()) / max(induced_from_Smir_area, 1e-300),
            }
        ]
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_mirror_average_source_detail.csv'
    summary_path = OUTDIR / 'chi_open_system_parity_contrast_diag_anchor_band_mirror_average_source_summary.csv'
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == '__main__':
    main()
