#!/usr/bin/env python3
"""
Extract a first-principles candidate closure profile for superradiant channel
normalization in the same 1D action-derived chain.

Given bound-state E_N from
  [-d^2/dz^2 + U(z;D)] psi_N = E_N psi_N,
we define channel-specific barrier actions with a centrifugal term:

  U_l(z;D) = U(z;D) + l(l+1)/(z^2 + eps^2),
  S_{N,l}(D) = \int_{z_-}^{z_+} dz sqrt(U_l - E_N).

A geometry-only channel-rate candidate is

  Gamma_{N,l}^{(geo)} = omega_N * exp(-2 S_{N,l}),

and the effective superradiant normalization profile is inferred by matching to
Gamma_{N,l} = A_l * omega_N * (omega_N M)^{4l+4}:

  A_l^{(fp)}(D;N) = Gamma_{N,l}^{(geo)} / [omega_N (omega_N M)^{4l+4}].

For scan diagnostics, we use normalized profiles at N_ref=2 and D_ref=12:

  Atilde_l(D) = A_l^{(fp)}(D;N_ref) / A_l^{(fp)}(D_ref;N_ref).

This is a conservative candidate benchmark and is NOT propagated into baseline
figures by default.
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
    M: float = 1.0


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


def eig_1d(U: np.ndarray, z: np.ndarray, n_states: int = 4) -> np.ndarray:
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


def solve_case(
    D: float,
    level: Level,
    zmax: float,
    p: PhysicalParams,
    n_ref: int,
) -> Dict[str, float]:
    z = np.linspace(-zmax, zmax, level.Nz)
    U = u_z(z, D, p)
    E = eig_1d(U, z, n_states=max(4, n_ref + 1))

    iref = n_ref - 1
    E_ref = float(E[iref])
    omega_ref2 = p.m0**2 + E_ref
    if omega_ref2 <= 0.0:
        raise RuntimeError(f"D={D}, level={level.name}: non-positive omega_ref^2")
    omega_ref = float(np.sqrt(omega_ref2))

    row: Dict[str, float] = {
        "D": D,
        "level": level.name,
        "Nz": level.Nz,
        "zmax": zmax,
        "N_ref": float(n_ref),
        "E_ref": E_ref,
        "omega_ref": omega_ref,
    }

    for ell in (1, 2):
        U_ell = U + (ell * (ell + 1)) / (z**2 + p.eps**2)
        S_ell = wkb_action_central(U_ell, z, E_ref)
        if not np.isfinite(S_ell):
            raise RuntimeError(f"D={D}, level={level.name}: invalid S_l for l={ell}")

        gamma_geo = omega_ref * np.exp(-2.0 * S_ell)
        gamma_sr_base = omega_ref * ((omega_ref * p.M) ** (4 * ell + 4))
        A_fp = gamma_geo / max(gamma_sr_base, 1e-300)

        row[f"S_l{ell}"] = float(S_ell)
        row[f"gamma_geo_l{ell}"] = float(gamma_geo)
        row[f"A{ell}_fp"] = float(A_fp)

    row["A21_ratio"] = row["A2_fp"] / max(row["A1_fp"], 1e-300)
    return row


def normalize_profiles(df: pd.DataFrame, d_ref: float) -> pd.DataFrame:
    out = df.copy()
    out["A1_profile"] = np.nan
    out["A2_profile"] = np.nan

    for lvl, sub in out.groupby("level"):
        ref_rows = sub[np.isclose(sub["D"], d_ref)]
        if ref_rows.empty:
            raise RuntimeError(f"Reference D={d_ref} missing in level={lvl}")
        A1_ref = float(ref_rows.iloc[0]["A1_fp"])
        A2_ref = float(ref_rows.iloc[0]["A2_fp"])

        mask = out["level"] == lvl
        out.loc[mask, "A1_profile"] = out.loc[mask, "A1_fp"] / A1_ref
        out.loc[mask, "A2_profile"] = out.loc[mask, "A2_fp"] / A2_ref

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract first-principles superradiant prefactor profiles from 1D chain.")
    ap.add_argument("--Ds", default="6,12,18", help="Comma-separated D values")
    ap.add_argument("--full-scan", action="store_true", help="Use D=4..20 integer grid (fine only)")
    ap.add_argument("--zmax", type=float, default=80.0)
    ap.add_argument("--D-ref", type=float, default=12.0)
    ap.add_argument("--N-ref", type=int, default=2)
    ap.add_argument("--outdir", default="/Users/boypatrick/codex/PSLT_20260202/output/superrad_fp_1d")
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
            rows.append(solve_case(D=D, level=lv, zmax=args.zmax, p=p, n_ref=args.N_ref))

    order_level = {"coarse": 0, "mid": 1, "fine": 2}
    df = pd.DataFrame(rows).sort_values(
        ["D", "level"], key=lambda s: s.map(order_level) if s.name == "level" else s
    )
    df = normalize_profiles(df, d_ref=args.D_ref)

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
                    "rel_A1_profile_vs_fine": abs(cur["A1_profile"] - fine["A1_profile"]) / max(abs(fine["A1_profile"]), 1e-30),
                    "rel_A2_profile_vs_fine": abs(cur["A2_profile"] - fine["A2_profile"]) / max(abs(fine["A2_profile"]), 1e-30),
                    "rel_A21_ratio_vs_fine": abs(cur["A21_ratio"] - fine["A21_ratio"]) / max(abs(fine["A21_ratio"]), 1e-30),
                }
            )
    rel = pd.DataFrame(rel_rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(int(d)) if float(d).is_integer() else str(d) for d in d_list)
    out_main = outdir / f"superrad_prefactor_D{tag}.csv"
    out_rel = outdir / f"superrad_prefactor_D{tag}_relerr.csv"
    df.to_csv(out_main, index=False)
    rel.to_csv(out_rel, index=False)

    print(f"[saved] {out_main}")
    print(f"[saved] {out_rel}")
    cols = ["D", "level", "omega_ref", "S_l1", "S_l2", "A1_fp", "A2_fp", "A1_profile", "A2_profile", "A21_ratio"]
    print(df[cols].to_string(index=False))
    if len(rel) > 0:
        print("\n[relative errors vs fine]")
        print(rel.to_string(index=False))


if __name__ == "__main__":
    main()
