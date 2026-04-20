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
    ap.add_argument("--rho-min-scan-min", type=float, default=0.03)
    ap.add_argument("--rho-min-scan-max", type=float, default=0.30)
    ap.add_argument("--rho-min-scan-step", type=float, default=0.03)
    ap.add_argument("--zeta-box-scan-min", type=float, default=1.10)
    ap.add_argument("--zeta-box-scan-max", type=float, default=1.20)
    ap.add_argument("--zeta-box-scan-step", type=float, default=0.01)
    ap.add_argument("--offaxis-rho-min-threshold", type=float, default=0.12)
    ap.add_argument("--canonical-rho-min", type=float, default=0.15)
    ap.add_argument("--canonical-zeta-box", type=float, default=1.20)
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

        N_mass_inner = float(np.sum(q_mass_neg[inner]))
        P_mass_inner = float(np.sum(q_mass_pos[inner]))
        cache.append((D, rr, local_zeta, inner, q_mass_neg, N_mass_inner, P_mass_inner))
        base_rows.append({"D": D, "N_mass_inner": N_mass_inner, "P_mass_inner": P_mass_inner})

    rho_min_values = np.arange(
        float(args.rho_min_scan_min),
        float(args.rho_min_scan_max) + 0.5 * float(args.rho_min_scan_step),
        float(args.rho_min_scan_step),
    )
    zeta_box_values = np.arange(
        float(args.zeta_box_scan_min),
        float(args.zeta_box_scan_max) + 0.5 * float(args.zeta_box_scan_step),
        float(args.zeta_box_scan_step),
    )

    scan_rows = []
    best_strict = None
    best_strict_detail = None
    best_offaxis = None
    best_offaxis_detail = None
    canonical_detail = None

    for rho_min in rho_min_values:
        for zeta_box in zeta_box_values:
            rows = []
            for D, rr, local_zeta, inner, q_mass_neg, N_mass_inner, P_mass_inner in cache:
                annulus = inner & (rr >= rho_min) & (rr <= float(args.rho_anchor)) & (local_zeta <= zeta_box)
                N_mass_annulus = float(np.sum(q_mass_neg[annulus]))
                N_mass_tail = N_mass_inner - N_mass_annulus
                rows.append(
                    {
                        "D": D,
                        "rho_min": float(rho_min),
                        "zeta_box": float(zeta_box),
                        "P_mass_inner": P_mass_inner,
                        "N_mass_inner": N_mass_inner,
                        "N_mass_annulus": N_mass_annulus,
                        "N_mass_tail": N_mass_tail,
                        "annulus_over_inner_share": N_mass_annulus / max(N_mass_inner, 1e-300),
                        "tail_over_P_mass_inner": N_mass_tail / max(P_mass_inner, 1e-300),
                        "annulus_over_P_mass_inner": N_mass_annulus / max(P_mass_inner, 1e-300),
                    }
                )
            df = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
            max_tail = float(df["tail_over_P_mass_inner"].max())
            min_share = float(df["annulus_over_inner_share"].min())
            mean_share = float(df["annulus_over_inner_share"].mean())
            max_annulus_over_P = float(df["annulus_over_P_mass_inner"].max())
            scan_rows.append(
                {
                    "rho_min": float(rho_min),
                    "zeta_box": float(zeta_box),
                    "max_tail_over_P_mass_inner": max_tail,
                    "min_annulus_over_inner_share": min_share,
                    "mean_annulus_over_inner_share": mean_share,
                    "max_annulus_over_P_mass_inner": max_annulus_over_P,
                }
            )
            cand = (-max_tail, min_share, mean_share, -float(rho_min), float(zeta_box))
            if best_strict is None or cand > best_strict:
                best_strict = cand
                best_strict_detail = df.copy()
            if float(rho_min) >= float(args.offaxis_rho_min_threshold):
                offaxis_cand = (-max_tail, min_share, mean_share, -float(rho_min), float(zeta_box))
                if best_offaxis is None or offaxis_cand > best_offaxis:
                    best_offaxis = offaxis_cand
                    best_offaxis_detail = df.copy()
            if abs(float(rho_min) - float(args.canonical_rho_min)) < 5e-13 and abs(float(zeta_box) - float(args.canonical_zeta_box)) < 5e-13:
                canonical_detail = df.copy()

    if best_strict_detail is None or best_offaxis_detail is None or canonical_detail is None:
        raise RuntimeError("Negative-carrier annulus scan did not produce the required profiles.")

    base = pd.DataFrame(base_rows).sort_values("D").reset_index(drop=True)
    n_floor_idx = int(base["N_mass_inner"].idxmax())
    best_strict_tail_idx = int(best_strict_detail["tail_over_P_mass_inner"].idxmax())
    best_offaxis_tail_idx = int(best_offaxis_detail["tail_over_P_mass_inner"].idxmax())
    canonical_tail_idx = int(canonical_detail["tail_over_P_mass_inner"].idxmax())
    canonical_share_idx = int(canonical_detail["annulus_over_inner_share"].idxmin())
    canonical_annulus_idx = int(canonical_detail["annulus_over_P_mass_inner"].idxmax())

    summary = pd.DataFrame(
        [
            {
                "N_mass_inner_max_abs": float(base["N_mass_inner"].max()),
                "N_mass_inner_max_D": float(base.loc[n_floor_idx, "D"]),
                "best_strict_rho_min": float(best_strict_detail.loc[0, "rho_min"]),
                "best_strict_zeta_box": float(best_strict_detail.loc[0, "zeta_box"]),
                "best_strict_max_tail_over_P_mass_inner": float(best_strict_detail["tail_over_P_mass_inner"].max()),
                "best_strict_max_tail_over_P_mass_inner_D": float(best_strict_detail.loc[best_strict_tail_idx, "D"]),
                "best_strict_min_annulus_over_inner_share": float(best_strict_detail["annulus_over_inner_share"].min()),
                "best_strict_min_annulus_over_inner_share_D": float(best_strict_detail.loc[best_strict_tail_idx, "D"]),
                "best_strict_max_annulus_over_P_mass_inner": float(best_strict_detail["annulus_over_P_mass_inner"].max()),
                "best_strict_max_annulus_over_P_mass_inner_D": float(
                    best_strict_detail.loc[int(best_strict_detail["annulus_over_P_mass_inner"].idxmax()), "D"]
                ),
                "best_offaxis_rho_min": float(best_offaxis_detail.loc[0, "rho_min"]),
                "best_offaxis_zeta_box": float(best_offaxis_detail.loc[0, "zeta_box"]),
                "best_offaxis_max_tail_over_P_mass_inner": float(best_offaxis_detail["tail_over_P_mass_inner"].max()),
                "best_offaxis_max_tail_over_P_mass_inner_D": float(best_offaxis_detail.loc[best_offaxis_tail_idx, "D"]),
                "best_offaxis_min_annulus_over_inner_share": float(best_offaxis_detail["annulus_over_inner_share"].min()),
                "best_offaxis_min_annulus_over_inner_share_D": float(
                    best_offaxis_detail.loc[int(best_offaxis_detail["annulus_over_inner_share"].idxmin()), "D"]
                ),
                "canonical_rho_min": float(args.canonical_rho_min),
                "canonical_zeta_box": float(args.canonical_zeta_box),
                "canonical_max_tail_over_P_mass_inner": float(canonical_detail["tail_over_P_mass_inner"].max()),
                "canonical_max_tail_over_P_mass_inner_D": float(canonical_detail.loc[canonical_tail_idx, "D"]),
                "canonical_min_annulus_over_inner_share": float(canonical_detail["annulus_over_inner_share"].min()),
                "canonical_min_annulus_over_inner_share_D": float(canonical_detail.loc[canonical_share_idx, "D"]),
                "canonical_max_annulus_over_P_mass_inner": float(canonical_detail["annulus_over_P_mass_inner"].max()),
                "canonical_max_annulus_over_P_mass_inner_D": float(canonical_detail.loc[canonical_annulus_idx, "D"]),
            }
        ]
    )
    summary["canonical_inner_max_over_annulus_max"] = (
        summary["N_mass_inner_max_abs"] / summary["canonical_max_annulus_over_P_mass_inner"] * 0.0
    )
    # Replace the dummy column with the intended ratio after the DataFrame exists.
    summary["canonical_inner_max_over_annulus_max"] = (
        summary["N_mass_inner_max_abs"]
        / float(base.loc[n_floor_idx, "P_mass_inner"])
        / summary["canonical_max_annulus_over_P_mass_inner"]
    )

    detail = base.merge(
        canonical_detail[["D", "N_mass_annulus", "N_mass_tail", "annulus_over_inner_share", "tail_over_P_mass_inner", "annulus_over_P_mass_inner"]],
        on="D",
        how="left",
    ).rename(
        columns={
            "N_mass_annulus": "canonical_N_mass_annulus",
            "N_mass_tail": "canonical_N_mass_tail",
            "annulus_over_inner_share": "canonical_annulus_over_inner_share",
            "tail_over_P_mass_inner": "canonical_tail_over_P_mass_inner",
            "annulus_over_P_mass_inner": "canonical_annulus_over_P_mass_inner",
        }
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_negative_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_negative_source_summary.csv"
    scan_path = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_negative_source_scan.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(scan_rows).sort_values(
        ["max_tail_over_P_mass_inner", "min_annulus_over_inner_share", "mean_annulus_over_inner_share", "rho_min", "zeta_box"],
        ascending=[True, False, False, True, False],
    ).to_csv(scan_path, index=False)
    print(detail_path)
    print(summary_path)
    print(scan_path)


if __name__ == "__main__":
    main()
