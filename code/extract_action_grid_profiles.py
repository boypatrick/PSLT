#!/usr/bin/env python3
"""
Build full scan-grid action-derived profiles for chi_LR(D) and A_l(D).

This script upgrades baseline inputs from sparse-knot interpolation to direct
action-derived extraction on the same D grid used by map scans.

Outputs:
  - output/chi_fp_2d/localized_chi_Dgrid{N}_fine.csv
  - output/superrad_fp_1d/superrad_prefactor_Dgrid{N}_fine.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from action_grid_profile_utils import scan_d_values
from extract_chi_localized_2d import Level as ChiLevel
from extract_chi_localized_2d import PhysicalParams as ChiParams
from extract_chi_localized_2d import run_case as run_chi_case
from extract_superrad_prefactor_1d import Level as SuperradLevel
from extract_superrad_prefactor_1d import PhysicalParams as SuperradParams
from extract_superrad_prefactor_1d import solve_case as run_superrad_case


ROOT = Path(__file__).resolve().parent.parent


def extract_chi_grid(
    d_vals: np.ndarray,
    outdir: Path,
    rho_max: float,
    z_margin: float,
    n_mu: int,
    tol: float,
    maxiter: int,
    sigma: float,
) -> Path:
    level = ChiLevel("fine", dr=0.06, dz=0.03)
    p = ChiParams()
    rows: List[dict] = []
    for i, d in enumerate(d_vals):
        if i % 10 == 0 or i == len(d_vals) - 1:
            print(f"[chi] {i+1}/{len(d_vals)} D={d:.6f}")
        row = run_chi_case(
            D=float(d),
            level=level,
            p=p,
            rho_max=float(rho_max),
            z_margin=float(z_margin),
            n_mu=int(n_mu),
            tol=float(tol),
            maxiter=int(maxiter),
            sigma=None if sigma < 0 else float(sigma),
        )
        row["grid_index"] = int(i)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"localized_chi_Dgrid{len(d_vals)}_fine.csv"
    df.to_csv(out, index=False)
    print(f"[saved] {out}")
    return out


def extract_superrad_grid(
    d_vals: np.ndarray,
    outdir: Path,
    zmax: float,
    d_ref: float,
    n_ref: int,
) -> Path:
    level = SuperradLevel("fine", Nz=8001)
    p = SuperradParams()

    ref = run_superrad_case(D=float(d_ref), level=level, zmax=float(zmax), p=p, n_ref=int(n_ref))
    a1_ref = float(ref["A1_fp"])
    a2_ref = float(ref["A2_fp"])

    rows: List[dict] = []
    for i, d in enumerate(d_vals):
        if i % 10 == 0 or i == len(d_vals) - 1:
            print(f"[A_l] {i+1}/{len(d_vals)} D={d:.6f}")
        row = run_superrad_case(D=float(d), level=level, zmax=float(zmax), p=p, n_ref=int(n_ref))
        row["grid_index"] = int(i)
        row["A1_profile"] = float(row["A1_fp"]) / max(a1_ref, 1e-300)
        row["A2_profile"] = float(row["A2_fp"]) / max(a2_ref, 1e-300)
        row["D_ref_profile"] = float(d_ref)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"superrad_prefactor_Dgrid{len(d_vals)}_fine.csv"
    df.to_csv(out, index=False)
    print(f"[saved] {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract full scan-grid action-derived chi_LR(D) and A_l(D) profiles.")
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=20.0)
    ap.add_argument("--d-num", type=int, default=60)
    ap.add_argument("--d-ref", type=float, default=12.0, help="Reference D used for A_l(D) profile normalization.")
    ap.add_argument("--n-ref", type=int, default=2, help="Reference mode index used in superradiant extraction.")
    ap.add_argument("--chi-rho-max", type=float, default=3.0)
    ap.add_argument("--chi-z-margin", type=float, default=6.0)
    ap.add_argument("--chi-n-mu", type=int, default=120)
    ap.add_argument("--chi-tol", type=float, default=1e-8)
    ap.add_argument("--chi-maxiter", type=int, default=30000)
    ap.add_argument("--chi-sigma", type=float, default=2.5)
    ap.add_argument("--superrad-zmax", type=float, default=80.0)
    args = ap.parse_args()

    d_vals = scan_d_values(args.d_min, args.d_max, args.d_num)
    print(
        "[grid]",
        f"D in [{args.d_min:.3f}, {args.d_max:.3f}],",
        f"N={args.d_num},",
        f"step={(d_vals[1] - d_vals[0]):.6f}" if len(d_vals) > 1 else "single point",
    )

    extract_chi_grid(
        d_vals=d_vals,
        outdir=ROOT / "output" / "chi_fp_2d",
        rho_max=args.chi_rho_max,
        z_margin=args.chi_z_margin,
        n_mu=args.chi_n_mu,
        tol=args.chi_tol,
        maxiter=args.chi_maxiter,
        sigma=args.chi_sigma,
    )
    extract_superrad_grid(
        d_vals=d_vals,
        outdir=ROOT / "output" / "superrad_fp_1d",
        zmax=args.superrad_zmax,
        d_ref=args.d_ref,
        n_ref=args.n_ref,
    )


if __name__ == "__main__":
    main()
