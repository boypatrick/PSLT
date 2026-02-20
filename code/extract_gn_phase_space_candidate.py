#!/usr/bin/env python3
"""
Preliminary 1D phase-space candidate extraction for g_N (not used in baseline scan).

This script evaluates a semiclassical candidate:

  rho_WKB(E) = (1 / 2pi) * integral dz / sqrt(E - U(z)),  on U(z) < E
  g_N^(ps)   = 1 + integral_{E_min}^{E_N} rho_WKB(E') dE'

where U(z)=V_eff(z)-m0^2 is constructed from the same action-derived geometry.
It is intended as a reproducibility tool for Appendix candidate checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def build_u_axis(z: np.ndarray, D: float, a: float, eps: float, m0: float, xi: float) -> np.ndarray:
    rp = np.sqrt((z - D / 2.0) ** 2 + eps**2)
    rm = np.sqrt((z + D / 2.0) ** 2 + eps**2)
    omega = 1.0 + a * (1.0 / rp + 1.0 / rm)
    lap_omega = -3.0 * a * eps**2 * ((rp**2 + eps**2) ** (-2.5) + (rm**2 + eps**2) ** (-2.5))
    v_eff = m0**2 * omega**2 + (1.0 - 6.0 * xi) * (lap_omega / omega)
    return v_eff - m0**2


def solve_bound_states_1d(z: np.ndarray, U: np.ndarray, n_states: int, sigma: float) -> np.ndarray:
    dz = z[1] - z[0]
    n = len(z)
    main = -2.0 * np.ones(n) / dz**2 + U
    off = np.ones(n - 1) / dz**2
    H = diags([off, main, off], [-1, 0, 1], format="csr")
    vals, _ = eigsh(H, k=max(6, n_states + 2), sigma=sigma, which="LM", tol=1e-10, maxiter=200000)
    vals = np.sort(np.real(vals))
    return vals[vals < 0][:n_states]


def rho_wkb(E: float, U: np.ndarray, z: np.ndarray) -> float:
    mask = U < E
    if np.count_nonzero(mask) < 4:
        return 0.0
    integrand = np.zeros_like(z)
    integrand[mask] = 1.0 / np.sqrt(np.maximum(E - U[mask], 1e-12))
    return simpson(integrand, x=z) / (2.0 * np.pi)


def g_phase_space(E_n: float, U: np.ndarray, z: np.ndarray, nE: int) -> float:
    E_min = float(np.min(U))
    if E_n <= E_min + 1e-8:
        return 1.0
    E_grid = np.linspace(E_min + 1e-6, E_n, nE)
    rho_vals = np.array([rho_wkb(E, U, z) for E in E_grid])
    return float(1.0 + simpson(rho_vals, x=E_grid))


def main() -> None:
    ap = argparse.ArgumentParser(description="Preliminary 1D phase-space candidate extraction for g_N.")
    ap.add_argument("--D", type=float, default=12.0)
    ap.add_argument("--zmax", type=float, default=80.0)
    ap.add_argument("--a", type=float, default=0.04)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--m0", type=float, default=1.0)
    ap.add_argument("--xi", type=float, default=0.14)
    ap.add_argument("--n-states", type=int, default=3)
    ap.add_argument("--nE", type=int, default=600)
    ap.add_argument("--sigma", type=float, default=-0.55)
    ap.add_argument("--outdir", default="/Users/boypatrick/codex/PSLT_20260202/output/gn_fp_1d")
    args = ap.parse_args()

    levels = [("coarse", 0.04), ("mid", 0.02), ("fine", 0.01)]
    rows = []

    for name, dz in levels:
        z = np.arange(-args.zmax, args.zmax + dz / 2.0, dz)
        U = build_u_axis(z, args.D, args.a, args.eps, args.m0, args.xi)
        Es = solve_bound_states_1d(z, U, n_states=args.n_states, sigma=args.sigma)
        g_vals = [g_phase_space(float(E), U, z, nE=args.nE) for E in Es]
        rows.append(
            {
                "level": name,
                "dz": dz,
                "Nz": len(z),
                "E1": float(Es[0]) if len(Es) > 0 else np.nan,
                "E2": float(Es[1]) if len(Es) > 1 else np.nan,
                "E3": float(Es[2]) if len(Es) > 2 else np.nan,
                "g1_ps": float(g_vals[0]) if len(g_vals) > 0 else np.nan,
                "g2_ps": float(g_vals[1]) if len(g_vals) > 1 else np.nan,
                "g3_ps": float(g_vals[2]) if len(g_vals) > 2 else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    fine = df[df["level"] == "fine"].iloc[0]
    for k in ["g1_ps", "g2_ps", "g3_ps"]:
        df[f"rel_{k}_vs_fine"] = np.abs(df[k] - fine[k]) / max(abs(float(fine[k])), 1e-30)
    df["max_rel_g_vs_fine"] = df[[f"rel_{k}_vs_fine" for k in ["g1_ps", "g2_ps", "g3_ps"]]].max(axis=1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"gn_phase_space_candidate_D{int(args.D)}.csv"
    df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print(f"\n[done] wrote {out_csv}")


if __name__ == "__main__":
    main()

