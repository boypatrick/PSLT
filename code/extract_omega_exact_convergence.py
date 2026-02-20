#!/usr/bin/env python3
"""
Extract bound-state omega_N convergence benchmarks for D={6,12,18}.

This script follows the same action-derived 1D on-axis operator chain used in
Appendix A:

  U(z) = m0^2 (Omega^2 - 1) + (1 - 6 xi) Omega^{-1} LapOmega
  [-d^2/dz^2 + U] psi = E psi,   E = omega^2 - m0^2

It produces a coarse/mid/fine convergence table for omega_1, omega_2 and
Delta omega_12 under the bound-state convention (E<0 and omega^2>0).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
    Ui = U[1:-1]  # Dirichlet boundaries
    Ni = len(Ui)

    main = 2.0 / dz**2 + Ui
    off = -1.0 / dz**2 * np.ones(Ni - 1)
    evals = la.eigh_tridiagonal(main, off, select="i", select_range=(0, n_states - 1))[0]
    return np.array(evals, dtype=float)


def solve_case(D: float, level: Level, zmax: float, p: PhysicalParams) -> Dict[str, float]:
    z = np.linspace(-zmax, zmax, level.Nz)
    U = u_z(z, D, p)
    E = eig_1d(U, z, n_states=3)

    omega2 = p.m0**2 + E
    omega = np.sqrt(np.maximum(omega2, 0.0))
    stable = (E < 0.0) & (omega2 > 0.0)

    # We report first two stable bound levels for benchmark consistency.
    out: Dict[str, float] = {
        "D": D,
        "level": level.name,
        "Nz": level.Nz,
        "zmax": zmax,
        "E1": float(E[0]),
        "E2": float(E[1]),
        "E3": float(E[2]),
        "omega1": float(omega[0]) if stable[0] else np.nan,
        "omega2": float(omega[1]) if stable[1] else np.nan,
        "stable1": bool(stable[0]),
        "stable2": bool(stable[1]),
    }
    out["delta_omega12"] = float(out["omega2"] - out["omega1"]) if np.isfinite(out["omega2"]) else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract omega_N convergence benchmarks (bound-state convention).")
    ap.add_argument("--Ds", default="6,12,18", help="Comma-separated D values.")
    ap.add_argument("--zmax", type=float, default=80.0, help="Half-box size in z.")
    ap.add_argument("--outdir", default="/Users/boypatrick/codex/PSLT_20260202/output/omega_fp_1d")
    args = ap.parse_args()

    d_list: List[float] = [float(x.strip()) for x in args.Ds.split(",") if x.strip()]
    levels = [
        Level("coarse", 4001),
        Level("mid", 6001),
        Level("fine", 8001),
    ]

    p = PhysicalParams()
    rows: List[Dict[str, float]] = []
    for D in d_list:
        for lv in levels:
            rows.append(solve_case(D=D, level=lv, zmax=args.zmax, p=p))

    df = pd.DataFrame(rows).sort_values(["D", "level"], key=lambda s: s.map({"coarse": 0, "mid": 1, "fine": 2}) if s.name == "level" else s)

    rel_rows: List[Dict[str, float]] = []
    for D in sorted(df["D"].unique()):
        sub = df[df["D"] == D].set_index("level")
        fine = sub.loc["fine"]
        for level in ["coarse", "mid", "fine"]:
            cur = sub.loc[level]
            rel_rows.append(
                {
                    "D": D,
                    "level": level,
                    "rel_omega1_vs_fine": abs(cur["omega1"] - fine["omega1"]) / max(abs(fine["omega1"]), 1e-30),
                    "rel_omega2_vs_fine": abs(cur["omega2"] - fine["omega2"]) / max(abs(fine["omega2"]), 1e-30),
                    "rel_delta_omega12_vs_fine": abs(cur["delta_omega12"] - fine["delta_omega12"]) / max(abs(fine["delta_omega12"]), 1e-30),
                }
            )

    rel = pd.DataFrame(rel_rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(int(d)) if float(d).is_integer() else str(d) for d in d_list)
    out_main = outdir / f"omega_exact_D{tag}.csv"
    out_rel = outdir / f"omega_exact_D{tag}_relerr.csv"
    df.to_csv(out_main, index=False)
    rel.to_csv(out_rel, index=False)

    print(f"[saved] {out_main}")
    print(f"[saved] {out_rel}")
    print(df[["D", "level", "E1", "E2", "omega1", "omega2", "delta_omega12"]].to_string(index=False))
    print("\n[relative errors vs fine]")
    print(rel.to_string(index=False))


if __name__ == "__main__":
    main()
