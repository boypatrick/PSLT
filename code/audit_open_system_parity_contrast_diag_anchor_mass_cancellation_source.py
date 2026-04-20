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


def _moment_parts(values: np.ndarray) -> tuple[float, float]:
    pos = float(np.sum(values[values > 0.0]))
    neg = float(np.sum(-values[values < 0.0]))
    return pos, neg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ds", default="4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    ap.add_argument("--rho-anchor", type=float, default=0.70)
    ap.add_argument("--zeta-anchor", type=float, default=1.21)
    ap.add_argument("--zeta-cut-min", type=float, default=0.80)
    ap.add_argument("--zeta-cut-max", type=float, default=1.20)
    ap.add_argument("--zeta-cut-step", type=float, default=0.01)
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

    fields_by_D: dict[float, dict[str, np.ndarray]] = {}
    base_rows = []
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
        fields_by_D[D] = f

        rr = f["rr"]
        zz = f["zz"]
        r_field = f["r_field"]
        C = f["contrast_density"]
        plus = (rr <= float(args.rho_anchor)) & (np.abs(zz - D / 2.0) <= float(args.zeta_anchor))
        minus = (rr <= float(args.rho_anchor)) & (np.abs(zz + D / 2.0) <= float(args.zeta_anchor))
        anchor = plus | minus

        diag_density = np.zeros_like(C)
        diag_density[plus] += C[plus] * f["delta_plus"][plus]
        diag_density[minus] += C[minus] * f["delta_minus"][minus]
        I_diag_anchor = float(np.sum(diag_density[anchor]))
        sign_diag = 1.0 if I_diag_anchor >= 0.0 else -1.0

        r_line = np.linspace(0.0, float(r_field.max()), 1800)
        interp = lambda arr: np.interp(r_field.ravel(), r_line, arr).reshape(r_field.shape)

        up_mass = _mass_field(rr, zz, D / 2.0, p)
        um_mass = _mass_field(rr, zz, -D / 2.0, p)
        up_der = _deriv_field(rr, zz, D / 2.0, p)
        um_der = _deriv_field(rr, zz, -D / 2.0, p)

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
        up_der_bar = spherical_average_generic(
            r_line,
            D,
            p,
            lambda rho, z, D, p: _deriv_field(rho, z, D / 2.0, p),
            120,
        )
        um_der_bar = spherical_average_generic(
            r_line,
            D,
            p,
            lambda rho, z, D, p: _deriv_field(rho, z, -D / 2.0, p),
            120,
        )

        delta_mass = np.zeros_like(C)
        delta_mass[plus] += C[plus] * (up_mass - interp(up_mass_bar))[plus]
        delta_mass[minus] += C[minus] * (um_mass - interp(um_mass_bar))[minus]
        delta_der = np.zeros_like(C)
        delta_der[plus] += C[plus] * (up_der - interp(up_der_bar))[plus]
        delta_der[minus] += C[minus] * (um_der - interp(um_der_bar))[minus]

        base_rows.append(
            {
                "D": D,
                "I_diag_anchor_abs": abs(I_diag_anchor),
                "sign_diag": sign_diag,
                "diag_density_sum_oriented": sign_diag * float(np.sum(diag_density[anchor])),
                "delta_mass_density": delta_mass,
                "delta_der_density": delta_der,
                "diag_density": diag_density,
            }
        )

    zeta_values = np.arange(
        float(args.zeta_cut_min),
        float(args.zeta_cut_max) + 0.5 * float(args.zeta_cut_step),
        float(args.zeta_cut_step),
    )
    scan_rows = []
    best = None
    best_zeta = None
    best_detail = None

    for zeta_cut in zeta_values:
        rows = []
        for row in base_rows:
            D = float(row["D"])
            f = fields_by_D[D]
            rr = f["rr"]
            zz = f["zz"]
            plus = (rr <= float(args.rho_anchor)) & (np.abs(zz - D / 2.0) <= float(args.zeta_anchor))
            minus = (rr <= float(args.rho_anchor)) & (np.abs(zz + D / 2.0) <= float(args.zeta_anchor))
            anchor = plus | minus

            zeta_p = np.abs(zz - D / 2.0)
            zeta_m = np.abs(zz + D / 2.0)
            inner = (plus & (zeta_p <= zeta_cut)) | (minus & (zeta_m <= zeta_cut))
            outer = anchor & (~inner)

            sign_diag = float(row["sign_diag"])
            oriented_mass_inner = sign_diag * row["delta_mass_density"][inner]
            P_mass_inner, N_mass_inner = _moment_parts(oriented_mass_inner)
            I_mass_inner = P_mass_inner - N_mass_inner

            I_der_inner = sign_diag * float(np.sum(row["delta_der_density"][inner]))
            I_outer = sign_diag * float(np.sum(row["diag_density"][outer]))
            I_diag_abs = float(row["I_diag_anchor_abs"])

            rows.append(
                {
                    "D": D,
                    "zeta_cut": float(zeta_cut),
                    "I_diag_anchor_abs": I_diag_abs,
                    "I_mass_inner": I_mass_inner,
                    "P_mass_inner": P_mass_inner,
                    "N_mass_inner": N_mass_inner,
                    "nu_diag_int_pm_m": N_mass_inner / max(P_mass_inner, 1e-300),
                    "I_der_inner": I_der_inner,
                    "I_outer": I_outer,
                    "adverse_derivative_loss": max(0.0, -I_der_inner),
                    "derivative_uplift": max(0.0, I_der_inner),
                    "adverse_outer_loss": max(0.0, -I_outer),
                    "outer_uplift": max(0.0, I_outer),
                    "split_resid_abs": abs(I_diag_abs - (I_mass_inner + I_der_inner + I_outer)),
                    "mass_inner_same_sign_as_diag": int(I_mass_inner >= -1e-18),
                    "lower_from_mass_and_adverse_losses": I_mass_inner - max(0.0, -I_der_inner) - max(0.0, -I_outer),
                    "adverse_derivative_over_P_mass_inner": max(0.0, -I_der_inner) / max(P_mass_inner, 1e-300),
                    "adverse_outer_over_P_mass_inner": max(0.0, -I_outer) / max(P_mass_inner, 1e-300),
                }
            )

        df = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
        mass_net_floor = float(df["I_mass_inner"].min())
        P_mass_floor = float(df["P_mass_inner"].min())
        max_nu_mass = float(df["nu_diag_int_pm_m"].max())
        scan_rows.append(
            {
                "zeta_cut": float(zeta_cut),
                "mass_inner_net_floor_abs": mass_net_floor,
                "P_mass_inner_floor_abs": P_mass_floor,
                "max_nu_diag_int_pm_m": max_nu_mass,
            }
        )
        cand = (mass_net_floor, P_mass_floor, -max_nu_mass, -float(zeta_cut))
        if best is None or cand > best:
            best = cand
            best_zeta = float(zeta_cut)
            best_detail = df.copy()

    if best_detail is None:
        raise RuntimeError("No interior mass-cancellation profile evaluated.")

    diag_floor_idx = int(best_detail["I_diag_anchor_abs"].idxmin())
    mass_floor_idx = int(best_detail["I_mass_inner"].idxmin())
    nu_idx = int(best_detail["nu_diag_int_pm_m"].idxmax())
    p_mass_floor_idx = int(best_detail["P_mass_inner"].idxmin())
    der_loss_idx = int(best_detail["adverse_derivative_over_P_mass_inner"].idxmax())
    out_loss_idx = int(best_detail["adverse_outer_over_P_mass_inner"].idxmax())

    summary = pd.DataFrame(
        [
            {
                "zeta_cut_opt": best_zeta,
                "max_split_resid_abs": float(best_detail["split_resid_abs"].max()),
                "all_mass_inner_same_sign_as_diag": int((best_detail["mass_inner_same_sign_as_diag"] == 1).all()),
                "diag_floor_abs": float(best_detail["I_diag_anchor_abs"].min()),
                "diag_floor_D": float(best_detail.loc[diag_floor_idx, "D"]),
                "mass_inner_net_floor_abs": float(best_detail["I_mass_inner"].min()),
                "mass_inner_net_floor_D": float(best_detail.loc[mass_floor_idx, "D"]),
                "max_nu_diag_int_pm_m": float(best_detail.loc[nu_idx, "nu_diag_int_pm_m"]),
                "max_nu_diag_int_pm_m_D": float(best_detail.loc[nu_idx, "D"]),
                "P_mass_inner_floor_abs": float(best_detail["P_mass_inner"].min()),
                "P_mass_inner_floor_D": float(best_detail.loc[p_mass_floor_idx, "D"]),
                "max_adverse_derivative_over_P_mass_inner": float(
                    best_detail.loc[der_loss_idx, "adverse_derivative_over_P_mass_inner"]
                ),
                "max_adverse_derivative_over_P_mass_inner_D": float(best_detail.loc[der_loss_idx, "D"]),
                "max_adverse_outer_over_P_mass_inner": float(
                    best_detail.loc[out_loss_idx, "adverse_outer_over_P_mass_inner"]
                ),
                "max_adverse_outer_over_P_mass_inner_D": float(best_detail.loc[out_loss_idx, "D"]),
                "num_derivative_uplift_knots": int((best_detail["derivative_uplift"] > 0.0).sum()),
                "num_adverse_derivative_knots": int((best_detail["adverse_derivative_loss"] > 0.0).sum()),
                "num_outer_uplift_knots": int((best_detail["outer_uplift"] > 0.0).sum()),
                "num_adverse_outer_knots": int((best_detail["adverse_outer_loss"] > 0.0).sum()),
                "derivative_uplift_at_mass_floor": float(best_detail.loc[mass_floor_idx, "derivative_uplift"]),
                "outer_uplift_at_mass_floor": float(best_detail.loc[mass_floor_idx, "outer_uplift"]),
                "adverse_derivative_loss_at_mass_floor": float(best_detail.loc[mass_floor_idx, "adverse_derivative_loss"]),
                "adverse_outer_loss_at_mass_floor": float(best_detail.loc[mass_floor_idx, "adverse_outer_loss"]),
                "mass_floor_aligns_with_diag_floor": int(
                    float(best_detail.loc[mass_floor_idx, "D"]) == float(best_detail.loc[diag_floor_idx, "D"])
                ),
            }
        ]
    )
    summary["diag_floor_over_mass_inner_net_floor"] = summary["diag_floor_abs"] / summary["mass_inner_net_floor_abs"]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_cancellation_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_cancellation_source_summary.csv"
    scan_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_cancellation_source_scan.csv"
    best_detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(scan_rows).sort_values("zeta_cut").to_csv(scan_path, index=False)
    print(detail_path)
    print(summary_path)
    print(scan_path)


if __name__ == "__main__":
    main()
