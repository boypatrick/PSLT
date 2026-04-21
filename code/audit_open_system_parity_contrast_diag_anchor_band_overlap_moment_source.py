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
        C = f["contrast_density"]
        psi1 = f["psi1"]
        psi2 = f["psi2"]

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

        measured_overlap = sign_diag * (psi1 * psi1 - psi2 * psi2) * level.dr * level.dz
        abs_overlap = np.abs(measured_overlap)
        pos_overlap = np.clip(measured_overlap, 0.0, None)
        neg_overlap = np.clip(-measured_overlap, 0.0, None)

        phiR = (psi1 + psi2) / math.sqrt(2.0)
        phiL = (psi1 - psi2) / math.sqrt(2.0)
        abs_overlap_from_localized = 2.0 * np.abs(phiR * phiL) * level.dr * level.dz

        M_band_neg = float(np.sum(neg_overlap[band]))
        M_band_pos = float(np.sum(pos_overlap[band]))
        M_band_abs = float(np.sum(abs_overlap[band]))
        M_band_signed = float(np.sum(measured_overlap[band]))
        localized_abs = float(np.sum(abs_overlap_from_localized[band]))
        neg_fraction = M_band_neg / max(M_band_abs, 1e-300)
        pos_fraction = M_band_pos / max(M_band_abs, 1e-300)
        signed_ratio = M_band_signed / max(M_band_abs, 1e-300)
        theta_neg = 0.5 * (1.0 - signed_ratio)
        identity_rhs = 0.5 * (M_band_abs - M_band_signed)

        rows.append(
            {
                "D": D,
                "M_band_neg": M_band_neg,
                "M_band_pos": M_band_pos,
                "M_band_abs": M_band_abs,
                "M_band_signed": M_band_signed,
                "neg_fraction": neg_fraction,
                "pos_fraction": pos_fraction,
                "signed_ratio": signed_ratio,
                "theta_neg": theta_neg,
                "identity_rhs": identity_rhs,
                "identity_resid_abs": abs(M_band_neg - identity_rhs),
                "localized_abs": localized_abs,
                "localized_abs_resid_abs": abs(M_band_abs - localized_abs),
            }
        )

    detail = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    neg_floor_idx = int(detail["M_band_neg"].idxmin())
    abs_floor_idx = int(detail["M_band_abs"].idxmin())
    signed_max_idx = int(detail["signed_ratio"].idxmax())
    signed_min_idx = int(detail["signed_ratio"].idxmin())
    theta_floor_idx = int(detail["theta_neg"].idxmin())

    theta_floor = float(detail["theta_neg"].min())
    induced_lower_bound = theta_floor * float(detail["M_band_abs"].min())

    summary = pd.DataFrame(
        [
            {
                "max_identity_resid_abs": float(detail["identity_resid_abs"].max()),
                "max_localized_abs_resid_abs": float(detail["localized_abs_resid_abs"].max()),
                "M_band_neg_floor_abs": float(detail["M_band_neg"].min()),
                "M_band_neg_floor_D": float(detail.loc[neg_floor_idx, "D"]),
                "M_band_abs_floor_abs": float(detail["M_band_abs"].min()),
                "M_band_abs_floor_D": float(detail.loc[abs_floor_idx, "D"]),
                "max_signed_ratio": float(detail["signed_ratio"].max()),
                "max_signed_ratio_D": float(detail.loc[signed_max_idx, "D"]),
                "min_signed_ratio": float(detail["signed_ratio"].min()),
                "min_signed_ratio_D": float(detail.loc[signed_min_idx, "D"]),
                "theta_neg_floor": theta_floor,
                "theta_neg_floor_D": float(detail.loc[theta_floor_idx, "D"]),
                "induced_lower_bound_abs": induced_lower_bound,
                "exact_over_induced_lower_bound": float(detail["M_band_neg"].min()) / max(induced_lower_bound, 1e-300),
            }
        ]
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_band_overlap_moment_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_band_overlap_moment_source_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
