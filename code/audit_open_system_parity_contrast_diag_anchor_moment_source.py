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
        C = f["contrast_density"]
        plus = (rr <= float(args.rho_anchor)) & (np.abs(zz - D / 2.0) <= float(args.zeta_anchor))
        minus = (rr <= float(args.rho_anchor)) & (np.abs(zz + D / 2.0) <= float(args.zeta_anchor))
        anchor = plus | minus

        diag_density = np.zeros_like(C)
        diag_density[plus] += C[plus] * f["delta_plus"][plus]
        diag_density[minus] += C[minus] * f["delta_minus"][minus]

        I_diag = float(np.sum(diag_density[anchor]))
        sign_diag = 1.0 if I_diag >= 0.0 else -1.0
        oriented = sign_diag * diag_density[anchor]
        positive = float(np.sum(oriented[oriented > 0.0]))
        negative = float(np.sum(-oriented[oriented < 0.0]))
        induced = positive - negative

        rows.append(
            {
                "D": D,
                "I_diag_anchor": abs(I_diag),
                "P_diag_anchor": positive,
                "N_diag_anchor": negative,
                "split_resid_abs": abs(abs(I_diag) - induced),
                "neg_over_pos": negative / max(positive, 1e-300),
                "pos_over_exact": positive / max(abs(I_diag), 1e-300),
                "neg_over_exact": negative / max(abs(I_diag), 1e-300),
                "induced_diag_lower_from_moments": induced,
            }
        )

    detail = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    diag_floor_idx = int(detail["I_diag_anchor"].idxmin())
    pos_floor_idx = int(detail["P_diag_anchor"].idxmin())
    ratio_idx = int(detail["neg_over_pos"].idxmax())
    induced_lower = float((1.0 - float(detail.loc[ratio_idx, "neg_over_pos"])) * float(detail.loc[pos_floor_idx, "P_diag_anchor"]))

    prev_mass = pd.read_csv(OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_source_summary.csv").iloc[0]
    prev_anchor = pd.read_csv(OUTDIR / "chi_open_system_parity_contrast_anchor_static_source_summary.csv").iloc[0]

    summary = pd.DataFrame(
        [
            {
                "max_split_resid_abs": float(detail["split_resid_abs"].max()),
                "diag_floor_abs": float(detail["I_diag_anchor"].min()),
                "diag_floor_D": float(detail.loc[diag_floor_idx, "D"]),
                "positive_moment_floor_abs": float(detail["P_diag_anchor"].min()),
                "positive_moment_floor_D": float(detail.loc[pos_floor_idx, "D"]),
                "max_neg_over_pos": float(detail.loc[ratio_idx, "neg_over_pos"]),
                "max_neg_over_pos_D": float(detail.loc[ratio_idx, "D"]),
                "induced_diag_lower_from_moments": induced_lower,
                "diag_floor_over_induced_moment_lower": float(detail["I_diag_anchor"].min()) / max(induced_lower, 1e-300),
                "mass_induced_lower": float(prev_mass["induced_diag_lower_from_mass_min"]),
                "moment_over_mass_induced": induced_lower / max(float(prev_mass["induced_diag_lower_from_mass_min"]), 1e-300),
                "anchor_induced_lower": float(prev_anchor["induced_anchor_lower_bound_canonical"]),
                "moment_over_anchor_induced": induced_lower / max(float(prev_anchor["induced_anchor_lower_bound_canonical"]), 1e-300),
            }
        ]
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_moment_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_moment_source_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
