#!/usr/bin/env python3
"""
Extract a first-principles candidate closure for eta from the same 1D
action-derived chain used for spectrum and WKB:

  DeltaE_12(D) ~ A_12(D) * exp(-S_1(D))
  A_12(D) = DeltaE_12(D) * exp(S_1(D))

Define dimensionless overlap-amplitude candidates (reference-normalized):

  eta_amp(D)  = A_12(D) / A_12(D_ref)
  eta_prob(D) = eta_amp(D)^2

This is a conservative candidate benchmark and is not forced into baseline scans.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.linalg as la


@dataclass(frozen=True)
class PhysicalParams:
    a: float = 0.04
    eps: float = 0.10
    m0: float = 1.0
    xi: float = 0.14


@dataclass(frozen=True)
class Level:
    name: str
    Nz: int


def omega_z(z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    r1 = np.sqrt((z - D / 2.0) ** 2 + p.eps**2)
    r2 = np.sqrt((z + D / 2.0) ** 2 + p.eps**2)
    return 1.0 + p.a * (1.0 / r1 + 1.0 / r2)


def laplace_omega_z(z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    r1 = np.sqrt((z - D / 2.0) ** 2 + p.eps**2)
    r2 = np.sqrt((z + D / 2.0) ** 2 + p.eps**2)
    lap1 = -3.0 * p.eps**2 / (r1**5)
    lap2 = -3.0 * p.eps**2 / (r2**5)
    return p.a * (lap1 + lap2)


def u_z(z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    om = omega_z(z, D, p)
    lap_om = laplace_omega_z(z, D, p)
    return p.m0**2 * (om**2 - 1.0) + (1.0 - 6.0 * p.xi) * (lap_om / om)


def eig_1d(U: np.ndarray, z: np.ndarray, n_states: int = 3) -> np.ndarray:
    dz = z[1] - z[0]
    Ui = U[1:-1]
    Ni = len(Ui)
    main = 2.0 / dz**2 + Ui
    off = -1.0 / dz**2 * np.ones(Ni - 1)
    evals = la.eigh_tridiagonal(main, off, select="i", select_range=(0, n_states - 1))[0]
    return np.array(evals, dtype=float)


def turning_points_near_zero(U: np.ndarray, z: np.ndarray, E: float) -> Optional[Tuple[float, float]]:
    y = U - E
    s = np.sign(y)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) < 2:
        return None

    roots = []
    for k in idx:
        z0, z1 = z[k], z[k + 1]
        y0, y1 = y[k], y[k + 1]
        zr = z0 - y0 * (z1 - z0) / (y1 - y0)
        roots.append(zr)
    roots = np.array(sorted(roots))

    neg = roots[roots < 0.0]
    pos = roots[roots > 0.0]
    if len(neg) == 0 or len(pos) == 0:
        return None
    return float(neg[-1]), float(pos[0])


def wkb_action_central(U: np.ndarray, z: np.ndarray, E: float) -> float:
    tp = turning_points_near_zero(U, z, E)
    if tp is None:
        return np.nan
    z1, z2 = tp
    y = U - E
    mask = (z >= z1) & (z <= z2) & (y > 0)
    if mask.sum() < 8:
        return np.nan
    return float(np.trapezoid(np.sqrt(y[mask]), z[mask]))


def solve_case(D: float, level: Level, zmax: float, p: PhysicalParams) -> Dict[str, float]:
    z = np.linspace(-zmax, zmax, level.Nz)
    U = u_z(z, D, p)
    E = eig_1d(U, z, n_states=3)

    omega2 = p.m0**2 + E
    omega = np.sqrt(np.maximum(omega2, 0.0))
    stable = (E < 0.0) & (omega2 > 0.0)
    if not (stable[0] and stable[1]):
        raise RuntimeError(f"D={D}, level={level.name}: first two states are not stable bound states.")

    E1 = float(E[0])
    E2 = float(E[1])
    deltaE = float(E2 - E1)
    S1 = wkb_action_central(U, z, E1)
    if not np.isfinite(S1):
        raise RuntimeError(f"D={D}, level={level.name}: failed to compute finite S1.")

    A_split = float(deltaE * np.exp(S1))
    return {
        "D": D,
        "level": level.name,
        "Nz": level.Nz,
        "zmax": zmax,
        "E1": E1,
        "E2": E2,
        "DeltaE12": deltaE,
        "S1": S1,
        "A_split": A_split,
        "omega1": float(omega[0]),
        "omega2": float(omega[1]),
    }


def normalize_eta(df: pd.DataFrame, d_ref: float) -> pd.DataFrame:
    out = df.copy()
    out["eta_amp"] = np.nan
    out["eta_prob"] = np.nan

    for lvl, sub in out.groupby("level"):
        ref_rows = sub[np.isclose(sub["D"], d_ref)]
        if ref_rows.empty:
            raise RuntimeError(f"Reference D={d_ref} missing in level={lvl}.")
        A_ref = float(ref_rows.iloc[0]["A_split"])
        mask = out["level"] == lvl
        out.loc[mask, "eta_amp"] = out.loc[mask, "A_split"] / A_ref
        out.loc[mask, "eta_prob"] = out.loc[mask, "eta_amp"] ** 2

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract eta first-principles prefactor candidates from 1D chain.")
    ap.add_argument("--Ds", default="6,12,18", help="Comma-separated D values.")
    ap.add_argument("--full-scan", action="store_true", help="Use D=4..20 integer grid (fine only).")
    ap.add_argument("--zmax", type=float, default=80.0)
    ap.add_argument("--D-ref", type=float, default=12.0, help="Reference D for eta normalization.")
    ap.add_argument("--outdir", default="/Users/boypatrick/codex/PSLT_20260202/output/eta_fp_1d")
    args = ap.parse_args()

    if args.full_scan:
        d_list: List[float] = [float(d) for d in range(4, 21)]
        levels = [Level("fine", 8001)]
    else:
        d_list = [float(x.strip()) for x in args.Ds.split(",") if x.strip()]
        levels = [Level("coarse", 4001), Level("mid", 6001), Level("fine", 8001)]

    p = PhysicalParams()
    rows: List[Dict[str, float]] = []
    for D in d_list:
        for lv in levels:
            rows.append(solve_case(D=D, level=lv, zmax=args.zmax, p=p))

    order_level = {"coarse": 0, "mid": 1, "fine": 2}
    df = pd.DataFrame(rows).sort_values(
        ["D", "level"], key=lambda s: s.map(order_level) if s.name == "level" else s
    )
    df = normalize_eta(df, d_ref=args.D_ref)

    rel_rows: List[Dict[str, float]] = []
    for D in sorted(df["D"].unique()):
        sub = df[df["D"] == D]
        if "fine" not in set(sub["level"]):
            continue
        fine = sub[sub["level"] == "fine"].iloc[0]
        for _, cur in sub.iterrows():
            rel_rows.append(
                {
                    "D": D,
                    "level": cur["level"],
                    "rel_S1_vs_fine": abs(cur["S1"] - fine["S1"]) / max(abs(fine["S1"]), 1e-30),
                    "rel_DeltaE12_vs_fine": abs(cur["DeltaE12"] - fine["DeltaE12"]) / max(abs(fine["DeltaE12"]), 1e-30),
                    "rel_eta_amp_vs_fine": abs(cur["eta_amp"] - fine["eta_amp"]) / max(abs(fine["eta_amp"]), 1e-30),
                    "rel_eta_prob_vs_fine": abs(cur["eta_prob"] - fine["eta_prob"]) / max(abs(fine["eta_prob"]), 1e-30),
                }
            )
    rel = pd.DataFrame(rel_rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(int(d)) if float(d).is_integer() else str(d) for d in d_list)
    out_main = outdir / f"eta_prefactor_D{tag}.csv"
    out_rel = outdir / f"eta_prefactor_D{tag}_relerr.csv"
    df.to_csv(out_main, index=False)
    rel.to_csv(out_rel, index=False)

    print(f"[saved] {out_main}")
    print(f"[saved] {out_rel}")
    show_cols = ["D", "level", "DeltaE12", "S1", "A_split", "eta_amp", "eta_prob"]
    print(df[show_cols].to_string(index=False))
    if len(rel) > 0:
        print("\n[relative errors vs fine]")
        print(rel.to_string(index=False))


if __name__ == "__main__":
    main()
