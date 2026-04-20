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
    ap.add_argument("--rho-corner-scan-min", type=float, default=0.22)
    ap.add_argument("--rho-corner-scan-max", type=float, default=0.30)
    ap.add_argument("--rho-corner-scan-step", type=float, default=0.01)
    ap.add_argument("--zeta-corner-scan-min", type=float, default=0.10)
    ap.add_argument("--zeta-corner-scan-max", type=float, default=0.40)
    ap.add_argument("--zeta-corner-scan-step", type=float, default=0.02)
    ap.add_argument("--canonical-rho-corner", type=float, default=0.22)
    ap.add_argument("--canonical-zeta-corner", type=float, default=0.10)
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

    cache = []
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

        rr = f["rr"]
        zz = f["zz"]
        C = f["contrast_density"]
        r_field = f["r_field"]
        plus = (rr <= float(args.rho_anchor)) & (np.abs(zz - D / 2.0) <= float(args.zeta_cut))
        minus = (rr <= float(args.rho_anchor)) & (np.abs(zz + D / 2.0) <= float(args.zeta_cut))
        inner = plus | minus

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

        delta_mass = np.zeros_like(C)
        delta_mass[plus] += C[plus] * (up_mass - interp(up_mass_bar))[plus]
        delta_mass[minus] += C[minus] * (um_mass - interp(um_mass_bar))[minus]

        q_mass = sign_diag * delta_mass
        q_mass_neg = np.clip(-q_mass, 0.0, None)
        q_mass_pos = np.clip(q_mass, 0.0, None)
        zeta_p = np.abs(zz - D / 2.0)
        zeta_m = np.abs(zz + D / 2.0)
        local_zeta = np.where(plus, zeta_p, zeta_m)

        band = inner & (rr >= float(args.rho_band)) & (rr <= float(args.rho_anchor)) & (local_zeta >= float(args.zeta_band_low)) & (local_zeta <= float(args.zeta_cut))
        N_mass_band = float(np.sum(q_mass_neg[band]))
        P_mass_inner = float(np.sum(q_mass_pos[inner]))

        cache.append((D, rr, local_zeta, band, q_mass_neg, N_mass_band, P_mass_inner))
        base_rows.append({"D": D, "N_mass_band": N_mass_band, "P_mass_inner": P_mass_inner})

    rho_values = np.arange(
        float(args.rho_corner_scan_min),
        float(args.rho_corner_scan_max) + 0.5 * float(args.rho_corner_scan_step),
        float(args.rho_corner_scan_step),
    )
    zeta_values = np.arange(
        float(args.zeta_corner_scan_min),
        float(args.zeta_corner_scan_max) + 0.5 * float(args.zeta_corner_scan_step),
        float(args.zeta_corner_scan_step),
    )

    scan_rows = []
    best = None
    best_detail = None
    canonical_detail = None
    metrics_rows = []
    for rho_corner in rho_values:
        for zeta_corner in zeta_values:
            rows = []
            for D, rr, local_zeta, band, q_mass_neg, N_mass_band, P_mass_inner in cache:
                sub = band & (rr >= float(rho_corner)) & (local_zeta >= float(zeta_corner))
                N_mass_corner = float(np.sum(q_mass_neg[sub]))
                N_mass_cap = N_mass_band - N_mass_corner
                rows.append(
                    {
                        "D": D,
                        "rho_corner": float(rho_corner),
                        "zeta_corner": float(zeta_corner),
                        "P_mass_inner": P_mass_inner,
                        "N_mass_band": N_mass_band,
                        "N_mass_corner": N_mass_corner,
                        "N_mass_cap": N_mass_cap,
                        "corner_over_band_share": N_mass_corner / max(N_mass_band, 1e-300),
                        "cap_over_P_mass_inner": N_mass_cap / max(P_mass_inner, 1e-300),
                        "corner_over_P_mass_inner": N_mass_corner / max(P_mass_inner, 1e-300),
                    }
                )
            df = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
            floor_abs = float(df["N_mass_corner"].min())
            min_share = float(df["corner_over_band_share"].min())
            mean_share = float(df["corner_over_band_share"].mean())
            max_cap = float(df["cap_over_P_mass_inner"].max())
            max_corner_over_P = float(df["corner_over_P_mass_inner"].max())
            scan_rows.append(
                {
                    "rho_corner": float(rho_corner),
                    "zeta_corner": float(zeta_corner),
                    "N_mass_corner_floor_abs": floor_abs,
                    "min_corner_over_band_share": min_share,
                    "mean_corner_over_band_share": mean_share,
                    "max_cap_over_P_mass_inner": max_cap,
                    "max_corner_over_P_mass_inner": max_corner_over_P,
                }
            )
            cand = (-max_cap, min_share, mean_share, floor_abs, -float(rho_corner), -float(zeta_corner))
            metrics_rows.append(
                {
                    "rho_corner": float(rho_corner),
                    "zeta_corner": float(zeta_corner),
                    "floor_abs": floor_abs,
                    "min_share": min_share,
                    "mean_share": mean_share,
                    "max_cap": max_cap,
                    "max_corner_over_P": max_corner_over_P,
                }
            )
            if best is None or cand > best:
                best = cand
                best_detail = df.copy()
            if abs(float(rho_corner) - float(args.canonical_rho_corner)) < 5e-13 and abs(float(zeta_corner) - float(args.canonical_zeta_corner)) < 5e-13:
                canonical_detail = df.copy()

    if best_detail is None or canonical_detail is None:
        raise RuntimeError("No strict adverse-corner profile evaluated.")

    metrics_df = pd.DataFrame(metrics_rows)
    tol = 1e-12
    best_max_cap = float(metrics_df["max_cap"].min())
    best_min_share = float(metrics_df.loc[np.abs(metrics_df["max_cap"] - best_max_cap) <= tol, "min_share"].max())
    best_mean_share = float(
        metrics_df.loc[
            (np.abs(metrics_df["max_cap"] - best_max_cap) <= tol)
            & (np.abs(metrics_df["min_share"] - best_min_share) <= tol),
            "mean_share",
        ].max()
    )
    best_floor = float(
        metrics_df.loc[
            (np.abs(metrics_df["max_cap"] - best_max_cap) <= tol)
            & (np.abs(metrics_df["min_share"] - best_min_share) <= tol)
            & (np.abs(metrics_df["mean_share"] - best_mean_share) <= tol),
            "floor_abs",
        ].max()
    )
    best_plateau = metrics_df.loc[
        (np.abs(metrics_df["max_cap"] - best_max_cap) <= tol)
        & (np.abs(metrics_df["min_share"] - best_min_share) <= tol)
        & (np.abs(metrics_df["mean_share"] - best_mean_share) <= tol)
        & (np.abs(metrics_df["floor_abs"] - best_floor) <= tol),
        ["rho_corner", "zeta_corner"],
    ]

    base = pd.DataFrame(base_rows).sort_values("D").reset_index(drop=True)
    band_floor_idx = int(base["N_mass_band"].idxmin())
    best_floor_idx = int(best_detail["N_mass_corner"].idxmin())
    best_cap_idx = int(best_detail["cap_over_P_mass_inner"].idxmax())
    canonical_floor_idx = int(canonical_detail["N_mass_corner"].idxmin())
    canonical_cap_idx = int(canonical_detail["cap_over_P_mass_inner"].idxmax())
    canonical_share_idx = int(canonical_detail["corner_over_band_share"].idxmin())
    canonical_corner_idx = int(canonical_detail["corner_over_P_mass_inner"].idxmax())

    summary = pd.DataFrame(
        [
            {
                "N_mass_band_floor_abs": float(base["N_mass_band"].min()),
                "N_mass_band_floor_D": float(base.loc[band_floor_idx, "D"]),
                "best_corner_rho_min": float(best_plateau["rho_corner"].min()),
                "best_corner_rho_max": float(best_plateau["rho_corner"].max()),
                "best_corner_zeta_min": float(best_plateau["zeta_corner"].min()),
                "best_corner_zeta_max": float(best_plateau["zeta_corner"].max()),
                "best_corner_floor_abs": float(best_detail["N_mass_corner"].min()),
                "best_corner_floor_D": float(best_detail.loc[best_floor_idx, "D"]),
                "best_corner_min_over_band_share": float(best_detail["corner_over_band_share"].min()),
                "best_corner_min_over_band_share_D": float(best_detail.loc[int(best_detail["corner_over_band_share"].idxmin()), "D"]),
                "best_corner_max_cap_over_P_mass_inner": float(best_detail["cap_over_P_mass_inner"].max()),
                "best_corner_max_cap_over_P_mass_inner_D": float(best_detail.loc[best_cap_idx, "D"]),
                "canonical_rho_corner": float(args.canonical_rho_corner),
                "canonical_zeta_corner": float(args.canonical_zeta_corner),
                "canonical_corner_floor_abs": float(canonical_detail["N_mass_corner"].min()),
                "canonical_corner_floor_D": float(canonical_detail.loc[canonical_floor_idx, "D"]),
                "canonical_min_corner_over_band_share": float(canonical_detail["corner_over_band_share"].min()),
                "canonical_min_corner_over_band_share_D": float(canonical_detail.loc[canonical_share_idx, "D"]),
                "canonical_max_cap_over_P_mass_inner": float(canonical_detail["cap_over_P_mass_inner"].max()),
                "canonical_max_cap_over_P_mass_inner_D": float(canonical_detail.loc[canonical_cap_idx, "D"]),
                "canonical_max_corner_over_P_mass_inner": float(canonical_detail["corner_over_P_mass_inner"].max()),
                "canonical_max_corner_over_P_mass_inner_D": float(canonical_detail.loc[canonical_corner_idx, "D"]),
            }
        ]
    )
    summary["band_floor_over_canonical_corner_floor"] = summary["N_mass_band_floor_abs"] / summary["canonical_corner_floor_abs"]

    detail = base.merge(
        canonical_detail[["D", "N_mass_corner", "N_mass_cap", "corner_over_band_share", "cap_over_P_mass_inner", "corner_over_P_mass_inner"]],
        on="D",
        how="left",
    ).rename(
        columns={
            "N_mass_corner": "canonical_N_mass_corner",
            "N_mass_cap": "canonical_N_mass_cap",
            "corner_over_band_share": "canonical_corner_over_band_share",
            "cap_over_P_mass_inner": "canonical_cap_over_P_mass_inner",
            "corner_over_P_mass_inner": "canonical_corner_over_P_mass_inner",
        }
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_band_corner_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_band_corner_source_summary.csv"
    scan_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_band_corner_source_scan.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(scan_rows).sort_values(
        ["max_cap_over_P_mass_inner", "min_corner_over_band_share", "mean_corner_over_band_share", "N_mass_corner_floor_abs", "rho_corner", "zeta_corner"],
        ascending=[True, False, False, False, True, True],
    ).to_csv(scan_path, index=False)
    print(detail_path)
    print(summary_path)
    print(scan_path)


if __name__ == "__main__":
    main()
