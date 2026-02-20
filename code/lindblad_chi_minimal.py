#!/usr/bin/env python3
"""
Minimal two-level Lindblad demonstrator for open-system chi diagnostics.

This script is for methodological exploration only. It does NOT replace the
baseline chi extraction in Appendix C and introduces environment rates that are
not yet derived from a microscopic EYMH bath model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def run_lindblad(
    delta: float,
    gamma_phi: float,
    gamma_mix: float,
    tmax: float,
    nstep: int,
):
    H = np.array([[0.0, delta / 2.0], [delta / 2.0, 0.0]], dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    L_ops = [np.sqrt(gamma_phi) * sigma_z, np.sqrt(gamma_mix) * sigma_x]

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        rho = y.reshape(2, 2)
        drho = -1j * (H @ rho - rho @ H)
        for L in L_ops:
            drho += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
        return drho.reshape(-1)

    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex).reshape(-1)
    t_eval = np.linspace(0.0, tmax, nstep)
    sol = solve_ivp(rhs, (0.0, tmax), rho0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

    rows = []
    for i, t in enumerate(sol.t):
        rho = sol.y[:, i].reshape(2, 2)
        rows.append(
            {
                "t": t,
                "rho_LL": float(np.real(rho[0, 0])),
                "rho_RR": float(np.real(rho[1, 1])),
                "abs_rho_LR": float(np.abs(rho[0, 1])),
            }
        )
    df = pd.DataFrame(rows)
    cmax = float(df["abs_rho_LR"].max())
    pmax = float(df["rho_RR"].max())
    return df, cmax, pmax


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal open-system chi demonstrator (not baseline).")
    ap.add_argument("--delta", type=float, default=0.358108, help="Level splitting Delta = lambda2-lambda1")
    ap.add_argument("--gamma-phi", type=float, default=0.05)
    ap.add_argument("--gamma-mix", type=float, default=0.01)
    ap.add_argument("--gamma-ref", type=float, default=808.7, help="Reference rate for optional dimensionless proxy")
    ap.add_argument("--tmax", type=float, default=200.0)
    ap.add_argument("--nstep", type=int, default=2000)
    ap.add_argument("--outdir", default="/Users/boypatrick/codex/PSLT_20260202/output/chi_open_system")
    args = ap.parse_args()

    df, cmax, pmax = run_lindblad(
        delta=args.delta,
        gamma_phi=args.gamma_phi,
        gamma_mix=args.gamma_mix,
        tmax=args.tmax,
        nstep=args.nstep,
    )
    # Optional normalized proxy (methodological only).
    chi_eff_proxy = float(2.0 * args.gamma_mix * cmax / max(args.gamma_ref, 1e-30))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "lindblad_minimal_timeseries.csv"
    df.to_csv(out_csv, index=False)

    print(f"Cmax = max_t |rho_LR| = {cmax:.6e}")
    print(f"Pmax = max_t rho_RR = {pmax:.6e}")
    print(f"chi_eff_proxy = 2*gamma_mix*Cmax/gamma_ref = {chi_eff_proxy:.6e}")
    print(f"[done] wrote {out_csv}")


if __name__ == "__main__":
    main()

