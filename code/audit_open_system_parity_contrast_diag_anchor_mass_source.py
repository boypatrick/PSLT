#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_open_system_parity_contrast_self_mirror_source import (
    solve_fields,
    omega_single,
    lap_omega_single,
    spherical_average_generic,
)
from extract_chi_localized_2d import Level, PhysicalParams

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def _mass_field(rho: np.ndarray, z: np.ndarray, center: float, p: PhysicalParams) -> np.ndarray:
    om = omega_single(rho, z, center, p)
    return p.m0 * p.m0 * (om * om - 1.0)


def _deriv_field(rho: np.ndarray, z: np.ndarray, center: float, p: PhysicalParams) -> np.ndarray:
    om = omega_single(rho, z, center, p)
    lap = lap_omega_single(rho, z, center, p)
    return (1.0 - 6.0 * p.xi) * (lap / om)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ds", default="4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    ap.add_argument("--rho-anchor", type=float, default=0.70)
    ap.add_argument("--zeta-anchor", type=float, default=1.21)
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
        r_field = f["r_field"]
        C = f["contrast_density"]
        plus_anchor = (rr <= float(args.rho_anchor)) & (np.abs(zz - D / 2.0) <= float(args.zeta_anchor))
        minus_anchor = (rr <= float(args.rho_anchor)) & (np.abs(zz + D / 2.0) <= float(args.zeta_anchor))

        r_line = np.linspace(0.0, float(r_field.max()), 1800)
        interp = lambda arr: np.interp(r_field.ravel(), r_line, arr).reshape(r_field.shape)

        up_mass = _mass_field(rr, zz, D / 2.0, p)
        um_mass = _mass_field(rr, zz, -D / 2.0, p)
        up_der = _deriv_field(rr, zz, D / 2.0, p)
        um_der = _deriv_field(rr, zz, -D / 2.0, p)

        up_mass_bar = spherical_average_generic(r_line, D, p, lambda rho, z, D, p: _mass_field(rho, z, D / 2.0, p), 120)
        um_mass_bar = spherical_average_generic(r_line, D, p, lambda rho, z, D, p: _mass_field(rho, z, -D / 2.0, p), 120)
        up_der_bar = spherical_average_generic(r_line, D, p, lambda rho, z, D, p: _deriv_field(rho, z, D / 2.0, p), 120)
        um_der_bar = spherical_average_generic(r_line, D, p, lambda rho, z, D, p: _deriv_field(rho, z, -D / 2.0, p), 120)

        delta_mass_p = up_mass - interp(up_mass_bar)
        delta_mass_m = um_mass - interp(um_mass_bar)
        delta_der_p = up_der - interp(up_der_bar)
        delta_der_m = um_der - interp(um_der_bar)

        I_diag = float(
            np.sum(C[plus_anchor] * f["delta_plus"][plus_anchor])
            + np.sum(C[minus_anchor] * f["delta_minus"][minus_anchor])
        )
        I_mass = float(
            np.sum(C[plus_anchor] * delta_mass_p[plus_anchor])
            + np.sum(C[minus_anchor] * delta_mass_m[minus_anchor])
        )
        I_der = float(
            np.sum(C[plus_anchor] * delta_der_p[plus_anchor])
            + np.sum(C[minus_anchor] * delta_der_m[minus_anchor])
        )
        sign_diag = 1.0 if I_diag >= 0.0 else -1.0
        opposite_loss = max(0.0, -sign_diag * I_der)
        same_uplift = max(0.0, sign_diag * I_der)

        rows.append(
            {
                "D": D,
                "I_diag_anchor": I_diag,
                "I_mass_anchor": I_mass,
                "I_der_anchor": I_der,
                "split_resid_abs": abs(I_diag - (I_mass + I_der)),
                "mass_same_sign_as_diag": int(I_mass == 0.0 or np.sign(I_mass) == sign_diag),
                "der_same_sign_as_diag": int(I_der == 0.0 or np.sign(I_der) == sign_diag),
                "opposite_derivative_loss": opposite_loss,
                "same_derivative_uplift": same_uplift,
                "opposite_loss_ratio_over_mass": opposite_loss / max(abs(I_mass), 1e-300),
                "same_uplift_ratio_over_mass": same_uplift / max(abs(I_mass), 1e-300),
                "mass_abs_share_of_diag": abs(I_mass) / max(abs(I_diag), 1e-300),
                "der_abs_share_of_diag": abs(I_der) / max(abs(I_diag), 1e-300),
                "induced_diag_lower_from_mass": abs(I_mass) - opposite_loss,
            }
        )

    detail = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    diag_floor_idx = int(detail["I_diag_anchor"].abs().idxmin())
    mass_floor_idx = int(detail["I_mass_anchor"].abs().idxmin())
    induced_idx = int(detail["induced_diag_lower_from_mass"].idxmin())
    summary = pd.DataFrame(
        [
            {
                "max_split_resid_abs": float(detail["split_resid_abs"].max()),
                "all_mass_same_sign_as_diag": int((detail["mass_same_sign_as_diag"] == 1).all()),
                "num_der_same_sign_as_diag": int((detail["der_same_sign_as_diag"] == 1).sum()),
                "diag_floor_abs": float(detail["I_diag_anchor"].abs().min()),
                "diag_floor_D": float(detail.loc[diag_floor_idx, "D"]),
                "mass_floor_abs": float(detail["I_mass_anchor"].abs().min()),
                "mass_floor_D": float(detail.loc[mass_floor_idx, "D"]),
                "max_opposite_loss_ratio_over_mass": float(detail["opposite_loss_ratio_over_mass"].max()),
                "max_opposite_loss_ratio_D": float(detail.loc[detail["opposite_loss_ratio_over_mass"].idxmax(), "D"]),
                "induced_diag_lower_from_mass_min": float(detail["induced_diag_lower_from_mass"].min()),
                "induced_diag_lower_from_mass_D": float(detail.loc[induced_idx, "D"]),
            }
        ]
    )
    summary["diag_floor_over_induced_mass_lower"] = summary["diag_floor_abs"] / summary["induced_diag_lower_from_mass_min"]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_source_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
