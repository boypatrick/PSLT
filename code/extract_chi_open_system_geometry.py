#!/usr/bin/env python3
"""
Geometry-informed open-system chi demonstrator (preliminary).

This script combines:
  - action-derived localized splitting data (M_LR, Gamma_ref),
  - a geometry-informed dephasing proxy gamma_phi from RMS(deltaV),
  - a two-level Lindblad evolution for diagnostic coherence amplitudes.

It is intended as a reproducible methodological bridge and is not used in the
baseline global scan statistics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse import diags, eye, kron

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHI_SOURCE = ROOT / "output" / "chi_fp_2d" / "localized_chi_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
DEFAULT_OUTDIR = ROOT / "output" / "chi_open_system"


@dataclass(frozen=True)
class PhysicalParams:
    a: float = 0.04
    eps: float = 0.1
    m0: float = 1.0
    xi: float = 0.14


def omega_2center(rho: np.ndarray, z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    rp2 = rho * rho + (z - D / 2.0) ** 2
    rm2 = rho * rho + (z + D / 2.0) ** 2
    return 1.0 + p.a * (1.0 / np.sqrt(rp2 + p.eps * p.eps) + 1.0 / np.sqrt(rm2 + p.eps * p.eps))


def lap_omega_2center(rho: np.ndarray, z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    rp2 = rho * rho + (z - D / 2.0) ** 2
    rm2 = rho * rho + (z + D / 2.0) ** 2
    return p.a * (-3.0 * p.eps * p.eps * ((rp2 + p.eps * p.eps) ** (-2.5) + (rm2 + p.eps * p.eps) ** (-2.5)))


def u_potential(rho: np.ndarray, z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    om = omega_2center(rho, z, D, p)
    lap_om = lap_omega_2center(rho, z, D, p)
    return p.m0 * p.m0 * (om * om - 1.0) + (1.0 - 6.0 * p.xi) * (lap_om / om)


def v_eff(rho: np.ndarray, z: np.ndarray, D: float, p: PhysicalParams) -> np.ndarray:
    return u_potential(rho, z, D, p) + p.m0 * p.m0


def spherical_average_v_eff(r_vals: np.ndarray, D: float, p: PhysicalParams, n_mu: int) -> np.ndarray:
    mu, w = np.polynomial.legendre.leggauss(n_mu)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    rr = r_vals[:, None]
    rho = rr * sin_theta[None, :]
    z = rr * mu[None, :]
    vv = v_eff(rho, z, D, p)
    return 0.5 * (vv @ w)


def build_grid(D: float, rho_max: float, z_margin: float, dr: float, dz: float, p: PhysicalParams):
    z_max = D / 2.0 + z_margin
    nr = int(round(rho_max / dr))
    nz = int(round(2.0 * z_max / dz))
    rho = (np.arange(nr) + 0.5) * dr
    z = -z_max + (np.arange(nz) + 0.5) * dz
    rr, zz = np.meshgrid(rho, z, indexing="ij")
    uu = u_potential(rr, zz, D, p)
    return rho, z, rr, zz, uu


def gamma_phi_from_geometry(D: float, p: PhysicalParams, rho_max: float, z_margin: float, dr: float, dz: float, n_mu: int) -> float:
    rho, z, rr, zz, uu = build_grid(D, rho_max, z_margin, dr, dz, p)
    vfull = uu + p.m0 * p.m0
    r_field = np.sqrt(rr * rr + zz * zz)
    r_line = np.linspace(0.0, float(r_field.max()), 1200)
    vbar_line = spherical_average_v_eff(r_line, D, p, n_mu=n_mu)
    vbar = np.interp(r_field.ravel(), r_line, vbar_line).reshape(r_field.shape)
    delta_v = vfull - vbar
    w = rho[:, None] * dr * dz
    num = np.sum(w * (delta_v**2))
    den = np.sum(w)
    return float(np.sqrt(max(num / max(den, 1e-30), 0.0)))


def lindblad_metrics(delta: float, gamma_phi: float, gamma_mix: float, tmax: float, nstep: int):
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
    Cmax = 0.0
    Pmax = 0.0
    for i in range(sol.y.shape[1]):
        rho = sol.y[:, i].reshape(2, 2)
        Cmax = max(Cmax, float(abs(rho[0, 1])))
        Pmax = max(Pmax, float(np.real(rho[1, 1])))
    return Cmax, Pmax


def main() -> None:
    ap = argparse.ArgumentParser(description="Geometry-informed open-system chi demonstrator.")
    ap.add_argument("--Ds", default="6,12,18")
    ap.add_argument("--rho-max", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.06)
    ap.add_argument("--dz", type=float, default=0.03)
    ap.add_argument("--n-mu", type=int, default=80)
    ap.add_argument("--tmax", type=float, default=200.0)
    ap.add_argument("--nstep", type=int, default=2000)
    ap.add_argument(
        "--chi-source",
        default=str(DEFAULT_CHI_SOURCE),
    )
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()

    D_list = [float(s.strip()) for s in args.Ds.split(",") if s.strip()]
    src = pd.read_csv(args.chi_source)
    src = src[src["level"] == "fine"].copy()
    p = PhysicalParams()

    rows = []
    for D in D_list:
        row_src = src[np.isclose(src["D"], D)]
        if row_src.empty:
            raise ValueError(f"Missing D={D} in {args.chi_source}")
        s = row_src.iloc[0]
        delta = float(s["E2"] - s["E1"])
        gamma_mix = float(s["M_LR"])
        gamma_ref = float(s["Gamma_ref"])
        chi_lr = float(s["chi_LR"])

        gamma_phi = gamma_phi_from_geometry(
            D=D,
            p=p,
            rho_max=args.rho_max,
            z_margin=args.z_margin,
            dr=args.dr,
            dz=args.dz,
            n_mu=args.n_mu,
        )

        Cmax, Pmax = lindblad_metrics(delta=delta, gamma_phi=gamma_phi, gamma_mix=gamma_mix, tmax=args.tmax, nstep=args.nstep)
        chi_eff_proxy = float(2.0 * gamma_mix * Cmax / max(gamma_ref, 1e-30))

        rows.append(
            {
                "D": D,
                "delta": delta,
                "gamma_phi_geom": gamma_phi,
                "gamma_mix_geom": gamma_mix,
                "Gamma_ref": gamma_ref,
                "Cmax": Cmax,
                "Pmax": Pmax,
                "chi_eff_proxy": chi_eff_proxy,
                "chi_LR_baseline": chi_lr,
                "ratio_proxy_to_baseline": chi_eff_proxy / max(chi_lr, 1e-30),
            }
        )

    out = pd.DataFrame(rows).sort_values("D").reset_index(drop=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(int(d)) for d in D_list)
    out_csv = outdir / f"chi_open_system_geometry_D{tag}.csv"
    out.to_csv(out_csv, index=False)

    print(out.to_string(index=False))
    print(f"\n[done] wrote {out_csv}")


if __name__ == "__main__":
    main()
