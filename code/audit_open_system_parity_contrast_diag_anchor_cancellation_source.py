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
        C = f["contrast_density"]
        r_field = f["r_field"]
        plus = (rr <= float(args.rho_anchor)) & (np.abs(zz - D / 2.0) <= float(args.zeta_anchor))
        minus = (rr <= float(args.rho_anchor)) & (np.abs(zz + D / 2.0) <= float(args.zeta_anchor))
        anchor = plus | minus

        diag_density = np.zeros_like(C)
        diag_density[plus] += C[plus] * f["delta_plus"][plus]
        diag_density[minus] += C[minus] * f["delta_minus"][minus]
        I_diag = float(np.sum(diag_density[anchor]))
        sign_diag = 1.0 if I_diag >= 0.0 else -1.0
        oriented = sign_diag * diag_density
        P_diag, N_diag = _moment_parts(oriented[anchor])

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

        mass_density = np.zeros_like(C)
        mass_density[plus] += C[plus] * (up_mass - interp(up_mass_bar))[plus]
        mass_density[minus] += C[minus] * (um_mass - interp(um_mass_bar))[minus]
        der_density = np.zeros_like(C)
        der_density[plus] += C[plus] * (up_der - interp(up_der_bar))[plus]
        der_density[minus] += C[minus] * (um_der - interp(um_der_bar))[minus]

        Pm, Nm = _moment_parts(sign_diag * mass_density[anchor])
        Pd, Nd = _moment_parts(sign_diag * der_density[anchor])

        base_rows.append(
            {
                "D": D,
                "I_diag_anchor": abs(I_diag),
                "P_diag_anchor": P_diag,
                "N_diag_anchor": N_diag,
                "nu_diag_pm": N_diag / max(P_diag, 1e-300),
                "P_mass": Pm,
                "N_mass": Nm,
                "P_der": Pd,
                "N_der": Nd,
                "mass_pos_over_total_pos": Pm / max(P_diag, 1e-300),
                "der_pos_over_total_pos": Pd / max(P_diag, 1e-300),
                "mass_neg_over_total_neg": Nm / max(N_diag, 1e-300),
                "der_neg_over_total_neg": Nd / max(N_diag, 1e-300),
            }
        )

    base = pd.DataFrame(base_rows).sort_values("D").reset_index(drop=True)

    zeta_values = np.arange(float(args.zeta_cut_min), float(args.zeta_cut_max) + 0.5 * float(args.zeta_cut_step), float(args.zeta_cut_step))
    scan_rows = []
    best = None
    best_zeta = None
    best_detail = None
    for zeta_cut in zeta_values:
        rows = []
        for D in Ds:
            f = fields_by_D[D]
            rr = f["rr"]
            zz = f["zz"]
            C = f["contrast_density"]
            plus = (rr <= float(args.rho_anchor)) & (np.abs(zz - D / 2.0) <= float(args.zeta_anchor))
            minus = (rr <= float(args.rho_anchor)) & (np.abs(zz + D / 2.0) <= float(args.zeta_anchor))
            anchor = plus | minus
            diag_density = np.zeros_like(C)
            diag_density[plus] += C[plus] * f["delta_plus"][plus]
            diag_density[minus] += C[minus] * f["delta_minus"][minus]
            I_diag = float(np.sum(diag_density[anchor]))
            sign_diag = 1.0 if I_diag >= 0.0 else -1.0
            oriented = sign_diag * diag_density

            zeta_p = np.abs(zz - D / 2.0)
            zeta_m = np.abs(zz + D / 2.0)
            inner = (plus & (zeta_p <= zeta_cut)) | (minus & (zeta_m <= zeta_cut))
            outer = anchor & (~inner)

            P_in, N_in = _moment_parts(oriented[inner])
            P_out, N_out = _moment_parts(oriented[outer])
            rows.append(
                {
                    "D": D,
                    "zeta_cut": zeta_cut,
                    "P_inner": P_in,
                    "N_inner": N_in,
                    "P_outer": P_out,
                    "N_outer": N_out,
                    "outer_neg_over_inner_pos": N_out / max(P_in, 1e-300),
                    "inner_neg_over_inner_pos": N_in / max(P_in, 1e-300),
                    "inner_pos_over_total_pos": P_in / max(float(base.loc[base['D'] == D, 'P_diag_anchor'].iloc[0]), 1e-300),
                }
            )
        df = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
        inner_pos_floor = float(df["P_inner"].min())
        max_outer_ratio = float(df["outer_neg_over_inner_pos"].max())
        max_inner_ratio = float(df["inner_neg_over_inner_pos"].max())
        induced = (1.0 - max_outer_ratio) * inner_pos_floor
        scan_rows.append(
            {
                "zeta_cut": float(zeta_cut),
                "inner_pos_floor_abs": inner_pos_floor,
                "max_outer_neg_over_inner_pos": max_outer_ratio,
                "max_inner_neg_over_inner_pos": max_inner_ratio,
                "induced_shell_isolated_lower": induced,
            }
        )
        cand = (induced, inner_pos_floor, -max_outer_ratio, -max_inner_ratio, -float(zeta_cut))
        if best is None or cand > best:
            best = cand
            best_zeta = float(zeta_cut)
            best_detail = df.copy()

    if best_detail is None:
        raise RuntimeError("No zeta-cut cancellation profile evaluated.")

    nu_idx = int(base["nu_diag_pm"].idxmax())
    pos_floor_idx = int(base["P_diag_anchor"].idxmin())
    mass_pos_idx = int(base["mass_pos_over_total_pos"].idxmin())
    der_neg_idx = int(base["der_neg_over_total_neg"].idxmax())
    mass_neg_idx = int(base["mass_neg_over_total_neg"].idxmin())

    best_detail_idx = int(best_detail["outer_neg_over_inner_pos"].idxmax())

    summary = pd.DataFrame(
        [
            {
                "diag_floor_abs": float(base["I_diag_anchor"].min()),
                "diag_floor_D": float(base.loc[int(base["I_diag_anchor"].idxmin()), "D"]),
                "nu_diag_pm": float(base.loc[nu_idx, "nu_diag_pm"]),
                "nu_diag_pm_D": float(base.loc[nu_idx, "D"]),
                "zeta_cut_opt": best_zeta,
                "inner_positive_floor_abs": float(best_detail["P_inner"].min()),
                "inner_positive_floor_D": float(best_detail.loc[int(best_detail["P_inner"].idxmin()), "D"]),
                "max_outer_neg_over_inner_pos": float(best_detail["outer_neg_over_inner_pos"].max()),
                "max_outer_neg_over_inner_pos_D": float(best_detail.loc[best_detail_idx, "D"]),
                "max_inner_neg_over_inner_pos": float(best_detail["inner_neg_over_inner_pos"].max()),
                "induced_shell_isolated_lower": float((1.0 - best_detail["outer_neg_over_inner_pos"].max()) * best_detail["P_inner"].min()),
                "min_mass_pos_over_total_pos": float(base.loc[mass_pos_idx, "mass_pos_over_total_pos"]),
                "min_mass_pos_over_total_pos_D": float(base.loc[mass_pos_idx, "D"]),
                "max_der_pos_over_total_pos": float(base["der_pos_over_total_pos"].max()),
                "max_der_pos_over_total_pos_D": float(base.loc[int(base["der_pos_over_total_pos"].idxmax()), "D"]),
                "min_mass_neg_over_total_neg": float(base.loc[mass_neg_idx, "mass_neg_over_total_neg"]),
                "min_mass_neg_over_total_neg_D": float(base.loc[mass_neg_idx, "D"]),
                "max_der_neg_over_total_neg": float(base.loc[der_neg_idx, "der_neg_over_total_neg"]),
                "max_der_neg_over_total_neg_D": float(base.loc[der_neg_idx, "D"]),
            }
        ]
    )
    summary["diag_floor_over_shell_isolated_lower"] = summary["diag_floor_abs"] / summary["induced_shell_isolated_lower"]

    detail = base.merge(best_detail, on="D", how="left")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_cancellation_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_cancellation_source_summary.csv"
    scan_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_cancellation_source_scan.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(scan_rows).sort_values("zeta_cut").to_csv(scan_path, index=False)
    print(detail_path)
    print(summary_path)
    print(scan_path)


if __name__ == "__main__":
    main()
