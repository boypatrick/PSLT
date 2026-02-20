#!/usr/bin/env python3
"""
Extract a first-principles dephasing candidate for t_coh from the same
action-derived 1D on-axis bound-state chain:

  [-d^2/dz^2 + U(z;D)] psi_n = E_n psi_n,
  E_n = omega_n^2 - m0^2,
  t_coh^(deph)(D) = pi / Delta_omega_12(D),
  Delta_omega_12 = omega_2 - omega_1.

This script is designed as a conservative replacement-candidate benchmark
and does not force baseline scan behavior.
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


def solve_case(D: float, level: Level, zmax: float, p: PhysicalParams, delta_floor: float) -> Dict[str, float]:
    z = np.linspace(-zmax, zmax, level.Nz)
    U = u_z(z, D, p)
    E = eig_1d(U, z, n_states=3)

    omega2 = p.m0**2 + E
    omega = np.sqrt(np.maximum(omega2, 0.0))
    stable = (E < 0.0) & (omega2 > 0.0)

    omega1 = float(omega[0]) if stable[0] else np.nan
    omega2v = float(omega[1]) if stable[1] else np.nan
    delta_omega12 = float(omega2v - omega1) if np.isfinite(omega2v) else np.nan
    delta_safe = max(abs(delta_omega12), delta_floor) if np.isfinite(delta_omega12) else np.nan
    t_coh_deph = float(np.pi / delta_safe) if np.isfinite(delta_safe) else np.nan

    return {
        "D": D,
        "level": level.name,
        "Nz": level.Nz,
        "zmax": zmax,
        "E1": float(E[0]),
        "E2": float(E[1]),
        "omega1": omega1,
        "omega2": omega2v,
        "delta_omega12": delta_omega12,
        "t_coh_deph": t_coh_deph,
        "stable1": bool(stable[0]),
        "stable2": bool(stable[1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract dephasing t_coh candidate from 1D on-axis chain.")
    ap.add_argument("--Ds", default="6,12,18", help="Comma-separated D values")
    ap.add_argument("--full-scan", action="store_true", help="Use D = 4..20 integer grid.")
    ap.add_argument("--zmax", type=float, default=80.0, help="Half-box size in z.")
    ap.add_argument("--delta-floor", type=float, default=1e-12, help="Safety floor for Delta omega in t_coh.")
    ap.add_argument("--outdir", default="/Users/boypatrick/codex/PSLT_20260202/output/tcoh_fp_1d")
    args = ap.parse_args()

    if args.full_scan:
        d_list: List[float] = [float(d) for d in range(4, 21)]
        levels = [Level("fine", 8001)]
    else:
        d_list = [float(x.strip()) for x in args.Ds.split(",") if x.strip()]
        levels = [
            Level("coarse", 4001),
            Level("mid", 6001),
            Level("fine", 8001),
        ]

    p = PhysicalParams()
    rows: List[Dict[str, float]] = []
    for D in d_list:
        for lv in levels:
            rows.append(solve_case(D=D, level=lv, zmax=args.zmax, p=p, delta_floor=args.delta_floor))

    order_level = {"coarse": 0, "mid": 1, "fine": 2}
    df = pd.DataFrame(rows).sort_values(
        ["D", "level"], key=lambda s: s.map(order_level) if s.name == "level" else s
    )

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
                    "rel_delta_omega12_vs_fine": abs(cur["delta_omega12"] - fine["delta_omega12"]) / max(abs(fine["delta_omega12"]), 1e-30),
                    "rel_tcoh_vs_fine": abs(cur["t_coh_deph"] - fine["t_coh_deph"]) / max(abs(fine["t_coh_deph"]), 1e-30),
                }
            )
    rel = pd.DataFrame(rel_rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(int(d)) if float(d).is_integer() else str(d) for d in d_list)
    out_main = outdir / f"tcoh_dephasing_D{tag}.csv"
    out_rel = outdir / f"tcoh_dephasing_D{tag}_relerr.csv"
    df.to_csv(out_main, index=False)
    rel.to_csv(out_rel, index=False)

    print(f"[saved] {out_main}")
    print(f"[saved] {out_rel}")
    cols = ["D", "level", "omega1", "omega2", "delta_omega12", "t_coh_deph"]
    print(df[cols].to_string(index=False))
    if len(rel) > 0:
        print("\n[relative errors vs fine]")
        print(rel.to_string(index=False))


if __name__ == "__main__":
    main()

