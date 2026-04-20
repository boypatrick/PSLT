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
    ap.add_argument("--rho-box-min", type=float, default=0.35)
    ap.add_argument("--rho-box-max", type=float, default=0.70)
    ap.add_argument("--rho-box-step", type=float, default=0.01)
    ap.add_argument("--zeta-box-min", type=float, default=0.70)
    ap.add_argument("--zeta-box-max", type=float, default=1.20)
    ap.add_argument("--zeta-box-step", type=float, default=0.01)
    ap.add_argument("--canonical-rho-box", type=float, default=0.70)
    ap.add_argument("--canonical-zeta-box", type=float, default=1.19)
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
    full_rows = []
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
        q_mass_pos = np.clip(sign_diag * delta_mass, 0.0, None)
        P_mass_inner = float(np.sum(q_mass_pos[inner]))

        cache.append((D, rr, zz, q_mass_pos, plus, minus, P_mass_inner))
        full_rows.append({"D": D, "P_mass_inner": P_mass_inner})

    rho_values = np.arange(
        float(args.rho_box_min),
        float(args.rho_box_max) + 0.5 * float(args.rho_box_step),
        float(args.rho_box_step),
    )
    zeta_values = np.arange(
        float(args.zeta_box_min),
        float(args.zeta_box_max) + 0.5 * float(args.zeta_box_step),
        float(args.zeta_box_step),
    )

    scan_rows = []
    best = None
    best_box = None
    best_detail = None
    canonical_detail = None
    for rho_box in rho_values:
        for zeta_box in zeta_values:
            rows = []
            for D, rr, zz, q_mass_pos, plus, minus, P_mass_inner in cache:
                region = (
                    (plus & (rr <= rho_box) & (np.abs(zz - D / 2.0) <= zeta_box))
                    | (minus & (rr <= rho_box) & (np.abs(zz + D / 2.0) <= zeta_box))
                )
                P_mass_box = float(np.sum(q_mass_pos[region]))
                rows.append(
                    {
                        "D": D,
                        "rho_box": float(rho_box),
                        "zeta_box": float(zeta_box),
                        "P_mass_inner": P_mass_inner,
                        "P_mass_box": P_mass_box,
                        "box_over_inner_share": P_mass_box / max(P_mass_inner, 1e-300),
                    }
                )
            df = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
            floor_abs = float(df["P_mass_box"].min())
            min_share = float(df["box_over_inner_share"].min())
            mean_share = float(df["box_over_inner_share"].mean())
            scan_rows.append(
                {
                    "rho_box": float(rho_box),
                    "zeta_box": float(zeta_box),
                    "P_mass_box_floor_abs": floor_abs,
                    "min_box_over_inner_share": min_share,
                    "mean_box_over_inner_share": mean_share,
                }
            )
            cand = (floor_abs, min_share, mean_share, float(rho_box), float(zeta_box))
            if best is None or cand > best:
                best = cand
                best_box = (float(rho_box), float(zeta_box))
                best_detail = df.copy()
            if abs(rho_box - float(args.canonical_rho_box)) < 5e-13 and abs(zeta_box - float(args.canonical_zeta_box)) < 5e-13:
                canonical_detail = df.copy()

    if best_detail is None:
        raise RuntimeError("No mass positive-carrier box profile evaluated.")
    if canonical_detail is None:
        raise RuntimeError("Canonical mass positive-carrier box was not scanned.")

    full = pd.DataFrame(full_rows).sort_values("D").reset_index(drop=True)
    full_floor_idx = int(full["P_mass_inner"].idxmin())
    canon_floor_idx = int(canonical_detail["P_mass_box"].idxmin())
    canon_share_idx = int(canonical_detail["box_over_inner_share"].idxmin())
    best_floor_idx = int(best_detail["P_mass_box"].idxmin())

    summary = pd.DataFrame(
        [
            {
                "P_mass_inner_floor_abs": float(full["P_mass_inner"].min()),
                "P_mass_inner_floor_D": float(full.loc[full_floor_idx, "D"]),
                "best_rho_box": float(best_box[0]),
                "best_zeta_box": float(best_box[1]),
                "best_box_floor_abs": float(best_detail["P_mass_box"].min()),
                "best_box_floor_D": float(best_detail.loc[best_floor_idx, "D"]),
                "best_box_min_share": float(best_detail["box_over_inner_share"].min()),
                "canonical_rho_box": float(args.canonical_rho_box),
                "canonical_zeta_box": float(args.canonical_zeta_box),
                "canonical_box_floor_abs": float(canonical_detail["P_mass_box"].min()),
                "canonical_box_floor_D": float(canonical_detail.loc[canon_floor_idx, "D"]),
                "canonical_min_box_over_inner_share": float(canonical_detail["box_over_inner_share"].min()),
                "canonical_min_box_over_inner_share_D": float(canonical_detail.loc[canon_share_idx, "D"]),
            }
        ]
    )
    summary["inner_floor_over_canonical_box_floor"] = summary["P_mass_inner_floor_abs"] / summary["canonical_box_floor_abs"]

    detail = full.merge(
        canonical_detail[["D", "P_mass_box", "box_over_inner_share"]],
        on="D",
        how="left",
    ).rename(
        columns={
            "P_mass_box": "canonical_P_mass_box",
            "box_over_inner_share": "canonical_box_over_inner_share",
        }
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_positive_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_positive_source_summary.csv"
    scan_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_positive_source_scan.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(scan_rows).sort_values(
        ["P_mass_box_floor_abs", "min_box_over_inner_share", "mean_box_over_inner_share", "rho_box", "zeta_box"],
        ascending=[False, False, False, False, False],
    ).to_csv(scan_path, index=False)
    print(detail_path)
    print(summary_path)
    print(scan_path)


if __name__ == "__main__":
    main()
