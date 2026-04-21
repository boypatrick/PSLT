#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_open_system_parity_contrast_self_mirror_source import (
    solve_fields,
    omega_single,
    spherical_average_generic,
)
from extract_chi_localized_2d import Level, PhysicalParams

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def _mass_field(rho: np.ndarray, z: np.ndarray, center: float, p: PhysicalParams) -> np.ndarray:
    om = omega_single(rho, z, center, p)
    return p.m0 * p.m0 * (om * om - 1.0)


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
        r_field = f["r_field"]

        plus = (rr <= float(args.rho_anchor)) & (np.abs(zz - D / 2.0) <= float(args.zeta_cut))
        minus = (rr <= float(args.rho_anchor)) & (np.abs(zz + D / 2.0) <= float(args.zeta_cut))
        inner = plus | minus
        zeta_p = np.abs(zz - D / 2.0)
        zeta_m = np.abs(zz + D / 2.0)
        local_zeta = np.where(plus, zeta_p, zeta_m)
        band = inner & (rr >= float(args.rho_band)) & (local_zeta >= float(args.zeta_band_low)) & (local_zeta <= float(args.zeta_cut))

        diag_density = np.zeros_like(C)
        diag_density[plus] += C[plus] * f["delta_plus"][plus]
        diag_density[minus] += C[minus] * f["delta_minus"][minus]
        sign_diag = 1.0 if float(np.sum(diag_density[inner])) >= 0.0 else -1.0

        r_line = np.linspace(0.0, float(r_field.max()), 1800)
        interp = lambda arr: np.interp(r_field.ravel(), r_line, arr).reshape(r_field.shape)

        up_mass = _mass_field(rr, zz, D / 2.0, p)
        um_mass = _mass_field(rr, zz, -D / 2.0, p)
        up_mass_bar = spherical_average_generic(
            r_line,
            D,
            p,
            lambda rho, z, D, p: _mass_field(rho, z, D / 2.0, p),
            120,
        )
        um_mass_bar = spherical_average_generic(
            r_line,
            D,
            p,
            lambda rho, z, D, p: _mass_field(rho, z, -D / 2.0, p),
            120,
        )

        local_mass = np.zeros_like(C)
        local_mass[plus] += (up_mass - interp(up_mass_bar))[plus]
        local_mass[minus] += (um_mass - interp(um_mass_bar))[minus]

        oriented_carrier = sign_diag * C
        negative_carrier = np.clip(-oriented_carrier, 0.0, None)
        band_kernel = local_mass[band]
        band_negative_carrier = negative_carrier[band]
        q_mass_neg = np.clip(-(oriented_carrier * local_mass), 0.0, None)[band]
        factored = band_kernel * band_negative_carrier

        N_mass_band = float(np.sum(q_mass_neg))
        carrier_neg_moment = float(np.sum(band_negative_carrier))
        kernel_min = float(np.min(band_kernel))
        kernel_max = float(np.max(band_kernel))
        kernel_mean = float(np.mean(band_kernel))
        kernel_avg_weighted = N_mass_band / max(carrier_neg_moment, 1e-300)
        lower_product = kernel_min * carrier_neg_moment
        upper_product = kernel_max * carrier_neg_moment
        factor_resid = float(np.max(np.abs(q_mass_neg - factored))) if q_mass_neg.size else 0.0
        positive_kernel_everywhere = int(kernel_min > 0.0)

        rows.append(
            {
                "D": D,
                "N_mass_band": N_mass_band,
                "carrier_neg_moment": carrier_neg_moment,
                "kernel_min": kernel_min,
                "kernel_mean": kernel_mean,
                "kernel_max": kernel_max,
                "kernel_avg_weighted": kernel_avg_weighted,
                "lower_product": lower_product,
                "upper_product": upper_product,
                "exact_over_lower": N_mass_band / max(lower_product, 1e-300),
                "upper_over_exact": upper_product / max(N_mass_band, 1e-300),
                "factor_resid_abs": factor_resid,
                "positive_kernel_everywhere": positive_kernel_everywhere,
            }
        )

    detail = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    kernel_min_idx = int(detail["kernel_min"].idxmin())
    carrier_floor_idx = int(detail["carrier_neg_moment"].idxmin())
    band_floor_idx = int(detail["N_mass_band"].idxmin())
    lower_floor_idx = int(detail["lower_product"].idxmin())
    avg_min_idx = int(detail["kernel_avg_weighted"].idxmin())
    avg_max_idx = int(detail["kernel_avg_weighted"].idxmax())
    exact_over_lower_idx = int(detail["exact_over_lower"].idxmax())
    upper_over_exact_idx = int(detail["upper_over_exact"].idxmax())

    summary = pd.DataFrame(
        [
            {
                "all_positive_kernel_everywhere": int(detail["positive_kernel_everywhere"].min()),
                "max_factor_resid_abs": float(detail["factor_resid_abs"].max()),
                "min_kernel_min": float(detail["kernel_min"].min()),
                "min_kernel_min_D": float(detail.loc[kernel_min_idx, "D"]),
                "min_weighted_kernel_avg": float(detail["kernel_avg_weighted"].min()),
                "min_weighted_kernel_avg_D": float(detail.loc[avg_min_idx, "D"]),
                "max_weighted_kernel_avg": float(detail["kernel_avg_weighted"].max()),
                "max_weighted_kernel_avg_D": float(detail.loc[avg_max_idx, "D"]),
                "carrier_neg_moment_floor_abs": float(detail["carrier_neg_moment"].min()),
                "carrier_neg_moment_floor_D": float(detail.loc[carrier_floor_idx, "D"]),
                "N_mass_band_floor_abs": float(detail["N_mass_band"].min()),
                "N_mass_band_floor_D": float(detail.loc[band_floor_idx, "D"]),
                "lower_product_floor_abs": float(detail["lower_product"].min()),
                "lower_product_floor_D": float(detail.loc[lower_floor_idx, "D"]),
                "max_exact_over_lower": float(detail["exact_over_lower"].max()),
                "max_exact_over_lower_D": float(detail.loc[exact_over_lower_idx, "D"]),
                "max_upper_over_exact": float(detail["upper_over_exact"].max()),
                "max_upper_over_exact_D": float(detail.loc[upper_over_exact_idx, "D"]),
                "band_floor_over_lower_product_floor": float(detail["N_mass_band"].min()) / max(float(detail["lower_product"].min()), 1e-300),
            }
        ]
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_band_magnitude_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_band_magnitude_source_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
