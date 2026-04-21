#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        C = f["contrast_density"]

        plus = (rr <= float(args.rho_anchor)) & (np.abs(zz - D / 2.0) <= float(args.zeta_cut))
        minus = (rr <= float(args.rho_anchor)) & (np.abs(zz + D / 2.0) <= float(args.zeta_cut))
        inner = plus | minus
        zeta_p = np.abs(zz - D / 2.0)
        zeta_m = np.abs(zz + D / 2.0)
        local_zeta = np.where(plus, zeta_p, zeta_m)
        band = inner & (rr >= float(args.rho_band)) & (local_zeta >= float(args.zeta_band_low)) & (local_zeta <= float(args.zeta_cut))

        diag = np.zeros_like(C)
        diag[plus] += C[plus] * f["delta_plus"][plus]
        diag[minus] += C[minus] * f["delta_minus"][minus]
        sign_diag = 1.0 if float(np.sum(diag[inner])) >= 0.0 else -1.0

        oriented_C = sign_diag * C
        W_band_neg_density = np.clip(-oriented_C, 0.0, None)
        negative_overlap_density = np.clip(-(oriented_C / (2.0 * np.pi * np.maximum(rr, 1e-300))), 0.0, None)

        W_band_neg = float(np.sum(W_band_neg_density[band]))
        M_band_neg = float(np.sum(negative_overlap_density[band]))
        rho_bar = W_band_neg / max(2.0 * np.pi * M_band_neg, 1e-300)
        factor_resid = float(np.max(np.abs(W_band_neg_density[band] - (2.0 * np.pi * rr[band] * negative_overlap_density[band])))) if np.any(band) else 0.0

        rows.append(
            {
                "D": D,
                "W_band_neg": W_band_neg,
                "M_band_neg": M_band_neg,
                "rho_bar_weighted": rho_bar,
                "factor_resid_abs": factor_resid,
                "rho_min_band": float(np.min(rr[band])),
                "rho_max_band": float(np.max(rr[band])),
                "W_over_M": W_band_neg / max(M_band_neg, 1e-300),
            }
        )

    detail = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    W_floor_idx = int(detail["W_band_neg"].idxmin())
    M_floor_idx = int(detail["M_band_neg"].idxmin())
    rho_bar_min_idx = int(detail["rho_bar_weighted"].idxmin())
    rho_bar_max_idx = int(detail["rho_bar_weighted"].idxmax())

    summary = pd.DataFrame(
        [
            {
                "max_factor_resid_abs": float(detail["factor_resid_abs"].max()),
                "W_band_neg_floor_abs": float(detail["W_band_neg"].min()),
                "W_band_neg_floor_D": float(detail.loc[W_floor_idx, "D"]),
                "M_band_neg_floor_abs": float(detail["M_band_neg"].min()),
                "M_band_neg_floor_D": float(detail.loc[M_floor_idx, "D"]),
                "min_rho_bar_weighted": float(detail["rho_bar_weighted"].min()),
                "min_rho_bar_weighted_D": float(detail.loc[rho_bar_min_idx, "D"]),
                "max_rho_bar_weighted": float(detail["rho_bar_weighted"].max()),
                "max_rho_bar_weighted_D": float(detail.loc[rho_bar_max_idx, "D"]),
                "rho_bar_relative_spread": (float(detail["rho_bar_weighted"].max()) - float(detail["rho_bar_weighted"].min())) / max(float(detail["rho_bar_weighted"].mean()), 1e-300),
                "W_floor_over_2pi_rho_bar_min_M_floor": float(detail["W_band_neg"].min()) / max(2.0 * np.pi * float(detail["rho_bar_weighted"].min()) * float(detail["M_band_neg"].min()), 1e-300),
            }
        ]
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_band_negative_carrier_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_band_negative_carrier_source_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
