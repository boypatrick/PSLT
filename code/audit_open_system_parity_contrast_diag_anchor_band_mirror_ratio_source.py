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
OUTDIR = ROOT / "output" / "chi_open_system"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ds", default="4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    ap.add_argument("--rho-anchor", type=float, default=0.70)
    ap.add_argument("--zeta-cut", type=float, default=1.20)
    ap.add_argument("--rho-band", type=float, default=0.20)
    ap.add_argument("--zeta-band-low", type=float, default=0.10)
    ap.add_argument("--rho-max-grid", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.06)
    ap.add_argument("--dz", type=float, default=0.03)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    args = ap.parse_args()

    Ds = [float(x.strip()) for x in args.Ds.split(",") if x.strip()]
    p = PhysicalParams()
    level = Level("fine", dr=float(args.dr), dz=float(args.dz))

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
        rr = f["rr"]
        zz = f["zz"]
        psi1 = f["psi1"]
        psi2 = f["psi2"]

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
        self_mean = S_self / max(area_band, 1e-300)
        mirror_mean = S_mir / max(area_band, 1e-300)
        mir_self_ratio = S_mir / max(S_self, 1e-300)
        factorized_mean = mir_self_ratio * self_mean

        rows.append(
            {
                "D": D,
                "A_band": area_band,
                "S_self": S_self,
                "S_mir": S_mir,
                "self_mean": self_mean,
                "mirror_mean": mirror_mean,
                "mir_self_ratio": mir_self_ratio,
                "factorized_mean": factorized_mean,
                "factor_resid_abs": abs(mirror_mean - factorized_mean),
            }
        )

    detail = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    mean_floor_idx = int(detail["mirror_mean"].idxmin())
    self_mean_floor_idx = int(detail["self_mean"].idxmin())
    ratio_floor_idx = int(detail["mir_self_ratio"].idxmin())
    ratio_ceil_idx = int(detail["mir_self_ratio"].idxmax())
    induced_lower = float(detail["self_mean"].min()) * float(detail["mir_self_ratio"].min())

    summary = pd.DataFrame(
        [
            {
                "max_factor_resid_abs": float(detail["factor_resid_abs"].max()),
                "mirror_mean_floor_abs": float(detail["mirror_mean"].min()),
                "mirror_mean_floor_D": float(detail.loc[mean_floor_idx, "D"]),
                "self_mean_floor_abs": float(detail["self_mean"].min()),
                "self_mean_floor_D": float(detail.loc[self_mean_floor_idx, "D"]),
                "mir_self_ratio_min": float(detail["mir_self_ratio"].min()),
                "mir_self_ratio_min_D": float(detail.loc[ratio_floor_idx, "D"]),
                "mir_self_ratio_max": float(detail["mir_self_ratio"].max()),
                "mir_self_ratio_max_D": float(detail.loc[ratio_ceil_idx, "D"]),
                "floors_align_same_D": int(
                    float(detail.loc[mean_floor_idx, "D"])
                    == float(detail.loc[self_mean_floor_idx, "D"])
                    == float(detail.loc[ratio_floor_idx, "D"])
                ),
                "induced_lower_bound_abs": induced_lower,
                "exact_over_induced_lower_bound": float(detail["mirror_mean"].min()) / max(induced_lower, 1e-300),
            }
        ]
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_band_mirror_ratio_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_band_mirror_ratio_source_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
