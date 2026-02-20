#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_D_axis1d_v3.py

Axis-1D scan engine for dual-center harmonic Omega -> derived U(z) potential.

Key high-standard changes vs v2:
  - Two presets:
      --preset diagnostic_deepwell : mechanism diagnostic (allows omega^2<0)
      --preset physical_gap        : stable bound modes (omega^2>0) for publication-grade interpretation
  - Outputs omega2, omega_real, omega_imag, has_stable_bound.
  - Physical tunneling is defined ONLY for stable bound modes:
        is_physical_tunnel = has_stable_bound AND turning points exist AND S>0
  - Adds optional convergence sweeps over Nz and zmax on representative D points.

Model (single-track derived):
  Veff = m0^2 Ω^2 + (1-6ξ) Ω^{-1} ∇^2Ω
  U    = Veff - m0^2 = m0^2(Ω^2-1) + (1-6ξ) Ω^{-1} ∇^2Ω
  Solve: [-d^2/dz^2 + U(z)] ψ = E ψ, where E = ω^2 - m0^2 and U(∞)=0.

NOTE:
  - Laplacian here uses the 3D identity on-axis as a fast scan engine.
  - For full paper, validate with 2D axisymmetric Laplacian in (rho,z) for a few D points.
"""

import argparse
import numpy as np
import pandas as pd
import scipy.linalg as la
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

# ------------------ Presets ------------------

@dataclass
class Preset:
    a: float
    eps: float
    m0: float
    xi: float
    zmax: float
    Nz: int
    nD: int
    nstates: int
    rep: str

PRESETS: Dict[str, Preset] = {
    # Diagnostic: extremely deep wells; will typically yield E<0 and omega^2<0 when m0=0.
    # Purpose: confirm double-well structure and monotonic S(D).
    "diagnostic_deepwell": Preset(
        a=8.0, eps=0.25, m0=0.0, xi=0.0,
        zmax=80.0, Nz=8001, nD=33, nstates=3,
        rep="6,12,18"
    ),
    # Physical: stable bound modes with omega^2 > 0; designed to keep E<0 while omega2 = m0^2+E > 0.
    # Purpose: publication-grade interpretation as spectrum + tunneling.
    "physical_gap": Preset(
        a=0.040, eps=0.10, m0=1.0, xi=0.14,
        zmax=80.0, Nz=8001, nD=33, nstates=3,
        rep="6,12,18"
    ),
}

# ------------------ Omega and Laplacian on axis ------------------

def omega_z(z: np.ndarray, D: float, a: float, eps: float) -> np.ndarray:
    r1 = np.sqrt((z - D/2.0)**2 + eps**2)
    r2 = np.sqrt((z + D/2.0)**2 + eps**2)
    return 1.0 + a*(1.0/r1 + 1.0/r2)

def laplace_omega_z(z: np.ndarray, D: float, a: float, eps: float) -> np.ndarray:
    """
    Fast on-axis diagnostic using 3D identity:
        ∇^2 (r^2+eps^2)^(-1/2) = -3 eps^2 (r^2+eps^2)^(-5/2)
    Strict 2D/3D axisymmetric validation should compute Laplacian in (rho,z).
    """
    r1 = np.sqrt((z - D/2.0)**2 + eps**2)
    r2 = np.sqrt((z + D/2.0)**2 + eps**2)
    lap1 = -3.0*eps**2/(r1**5)
    lap2 = -3.0*eps**2/(r2**5)
    return a*(lap1 + lap2)

def U_z(z: np.ndarray, D: float, a: float, eps: float, m0: float, xi: float) -> np.ndarray:
    Om = omega_z(z, D, a, eps)
    lapOm = laplace_omega_z(z, D, a, eps)
    # U = m0^2(Ω^2-1) + (1-6ξ) Ω^{-1} ∇^2Ω
    return m0**2*(Om**2 - 1.0) + (1.0 - 6.0*xi)*(lapOm/Om)

# ------------------ 1D eigen-solver ------------------

def eig_1d(U: np.ndarray, z: np.ndarray, n_states: int = 3) -> Tuple[np.ndarray, List[np.ndarray]]:
    dz = z[1] - z[0]
    Ui = U[1:-1]  # Dirichlet boundary
    Ni = len(Ui)

    main = 2.0/dz**2 + Ui
    off  = -1.0/dz**2 * np.ones(Ni-1)

    E, V = la.eigh_tridiagonal(main, off)
    E = E[:n_states]

    vecs: List[np.ndarray] = []
    for k in range(len(E)):
        v = np.zeros_like(z)
        v[1:-1] = V[:, k]
        norm = np.sqrt(np.trapz(v*v, z))
        if norm > 0:
            v = v / norm
        vecs.append(v)
    return E, vecs

# ------------------ Turning points and WKB ------------------

def turning_points_near_zero(U: np.ndarray, z: np.ndarray, E: float) -> Optional[Tuple[float, float]]:
    y = U - E
    s = np.sign(y)
    idx = np.where(s[:-1]*s[1:] < 0)[0]
    if len(idx) < 2:
        return None

    roots = []
    for k in idx:
        z0, z1 = z[k], z[k+1]
        y0, y1 = y[k], y[k+1]
        zr = z0 - y0*(z1-z0)/(y1-y0)
        roots.append(zr)
    roots = np.array(sorted(roots))

    neg = roots[roots < 0.0]
    pos = roots[roots > 0.0]
    if len(neg) == 0 or len(pos) == 0:
        return None
    return float(neg[-1]), float(pos[0])  # closest pair across 0

def wkb_action_central(U: np.ndarray, z: np.ndarray, E: float) -> Tuple[float, Optional[Tuple[float, float]]]:
    tp = turning_points_near_zero(U, z, E)
    if tp is None:
        return np.nan, None
    z1, z2 = tp
    y = U - E
    mask = (z >= z1) & (z <= z2) & (y > 0)
    if mask.sum() < 8:
        return np.nan, (z1, z2)
    S = np.trapz(np.sqrt(y[mask]), z[mask])
    return float(S), (z1, z2)

# ------------------ Scan core ------------------

def scan_D(
    D_list: np.ndarray,
    a: float, eps: float, m0: float, xi: float,
    zmax: float, Nz: int,
    n_states: int,
    reps: List[float]
) -> Tuple[pd.DataFrame, Dict[float, Dict[str, np.ndarray]]]:

    z = np.linspace(-zmax, zmax, Nz)
    rows = []
    rep_data: Dict[float, Dict[str, np.ndarray]] = {}

    for D in D_list:
        U = U_z(z, float(D), a, eps, m0, xi)
        Evals, vecs = eig_1d(U, z, n_states=n_states)

        for n, E in enumerate(Evals, start=1):
            omega2 = m0**2 + float(E)
            has_bound = (E < 0.0)
            has_stable_bound = bool(has_bound and (omega2 > 0.0))

            # omega decomposition
            omega_real = np.sqrt(omega2) if omega2 > 0 else np.nan
            omega_imag = np.sqrt(-omega2) if omega2 < 0 else 0.0

            if has_stable_bound:
                S, tp = wkb_action_central(U, z, float(E))
            else:
                S, tp = (np.nan, None)

            has_tp = (tp is not None)
            is_phys_tunnel = bool(has_stable_bound and has_tp and np.isfinite(S) and (S > 0.0))

            rows.append({
                "D": float(D), "n": int(n),
                "E": float(E),
                "omega2": float(omega2),
                "omega_real": float(omega_real) if np.isfinite(omega_real) else np.nan,
                "omega_imag": float(omega_imag),
                "S_z": float(S) if np.isfinite(S) else np.nan,
                "z1": None if tp is None else float(tp[0]),
                "z2": None if tp is None else float(tp[1]),
                "has_bound": bool(has_bound),
                "has_stable_bound": bool(has_stable_bound),
                "has_turning_points": bool(has_tp),
                "is_physical_tunnel": bool(is_phys_tunnel),
            })

        if any(abs(float(D) - rD) < 1e-12 for rD in reps):
            rep_data[float(D)] = {
                "z": z.copy(),
                "U": U.copy(),
                "E": Evals.copy(),
                # stack psi_n as columns
                "psi": np.vstack(vecs) if len(vecs) > 0 else np.empty((0, len(z))),
            }

    return pd.DataFrame(rows), rep_data

def save_representatives(rep_data: Dict[float, Dict[str, np.ndarray]], prefix: str) -> None:
    for D, pack in rep_data.items():
        z = pack["z"]
        U = pack["U"]
        E = pack["E"]
        psi = pack["psi"]  # shape: (n_states, Nz)

        pd.DataFrame({"z": z, "U": U}).to_csv(f"{prefix}_U_D{D:g}.csv", index=False)

        cols = {"z": z}
        for i in range(psi.shape[0]):
            cols[f"psi_{i+1}"] = psi[i, :]
        pd.DataFrame(cols).to_csv(f"{prefix}_psi_D{D:g}.csv", index=False)

        pd.DataFrame({"D": [D]*len(E), "n": np.arange(1, len(E)+1), "E": E}).to_csv(
            f"{prefix}_meta_D{D:g}.csv", index=False
        )

# ------------------ Convergence utility ------------------

def convergence_sweep(
    D_points: List[float],
    a: float, eps: float, m0: float, xi: float,
    zmax_list: List[float],
    Nz_list: List[int],
    n_states: int
) -> pd.DataFrame:
    """
    For each D in D_points, sweep (zmax, Nz) and record E1 and S1 (stable bound only).
    """
    rows = []
    for D in D_points:
        for zmax in zmax_list:
            for Nz in Nz_list:
                z = np.linspace(-zmax, zmax, Nz)
                U = U_z(z, D, a, eps, m0, xi)
                Evals, _ = eig_1d(U, z, n_states=n_states)
                E1 = float(Evals[0])
                omega2_1 = m0**2 + E1
                stable1 = (E1 < 0.0) and (omega2_1 > 0.0)
                if stable1:
                    S1, tp = wkb_action_central(U, z, E1)
                else:
                    S1, tp = (np.nan, None)
                rows.append({
                    "D": D,
                    "zmax": float(zmax),
                    "Nz": int(Nz),
                    "E1": E1,
                    "omega2_1": float(omega2_1),
                    "stable1": bool(stable1),
                    "S1": float(S1) if np.isfinite(S1) else np.nan,
                    "has_tp1": bool(tp is not None),
                })
    df = pd.DataFrame(rows)
    return df

# ------------------ CLI ------------------

def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, default="physical_gap",
                    choices=list(PRESETS.keys()),
                    help="Parameter preset: diagnostic_deepwell or physical_gap")

    ap.add_argument("--Dmin", type=float, default=4.0)
    ap.add_argument("--Dmax", type=float, default=20.0)
    ap.add_argument("--nD", type=int, default=None)

    ap.add_argument("--a", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--m0", type=float, default=None)
    ap.add_argument("--xi", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)
    ap.add_argument("--Nz", type=int, default=None)
    ap.add_argument("--nstates", type=int, default=None)
    ap.add_argument("--rep", type=str, default=None)

    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--rep_prefix", type=str, default="rep_axis1d_v3")

    ap.add_argument("--convergence", action="store_true",
                    help="Run convergence sweep instead of full scan.")
    ap.add_argument("--conv_D", type=str, default="6,12,18")
    ap.add_argument("--conv_zmax", type=str, default="60,80,100")
    ap.add_argument("--conv_Nz", type=str, default="3001,5001,8001")
    ap.add_argument("--conv_out", type=str, default="convergence_axis1d_v3.csv")

    args = ap.parse_args()

    P = PRESETS[args.preset]

    # Apply preset defaults unless overridden
    nD = args.nD if args.nD is not None else P.nD
    a = args.a if args.a is not None else P.a
    eps = args.eps if args.eps is not None else P.eps
    m0 = args.m0 if args.m0 is not None else P.m0
    xi = args.xi if args.xi is not None else P.xi
    zmax = args.zmax if args.zmax is not None else P.zmax
    Nz = args.Nz if args.Nz is not None else P.Nz
    nstates = args.nstates if args.nstates is not None else P.nstates
    rep = args.rep if args.rep is not None else P.rep
    reps = parse_list_floats(rep)

    if args.out is None:
        out = f"scan_axis1d_v3_{args.preset}.csv"
    else:
        out = args.out

    if args.convergence:
        D_points = parse_list_floats(args.conv_D)
        zmax_list = parse_list_floats(args.conv_zmax)
        Nz_list = parse_list_ints(args.conv_Nz)
        dfc = convergence_sweep(D_points, a, eps, m0, xi, zmax_list, Nz_list, nstates)
        dfc.to_csv(args.conv_out, index=False)
        print(f"[convergence] Saved: {args.conv_out}")
        print(dfc.head(12))
        return

    D_list = np.linspace(args.Dmin, args.Dmax, nD)
    df, rep_data = scan_D(D_list, a, eps, m0, xi, zmax, Nz, nstates, reps)
    df.to_csv(out, index=False)
    save_representatives(rep_data, prefix=args.rep_prefix)

    # Quick reviewer-style summary
    nb = int(df["has_bound"].sum())
    nsb = int(df["has_stable_bound"].sum())
    nt = int(df["is_physical_tunnel"].sum())
    print(f"Saved scan: {out}")
    print(f"Saved representatives: {args.rep_prefix}_U_D*.csv, {args.rep_prefix}_psi_D*.csv, {args.rep_prefix}_meta_D*.csv")
    print(f"Summary rows={len(df)} | has_bound={nb} | has_stable_bound={nsb} | is_physical_tunnel={nt}")
    if nsb > 0:
        print(df[df["has_stable_bound"]].head(10))
    else:
        print("No stable bound modes found (omega2<=0 for E<0). Try preset physical_gap or adjust (a,eps,xi,m0).")

if __name__ == "__main__":
    main()

