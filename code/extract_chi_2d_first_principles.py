#!/usr/bin/env python3
"""
Extract chi from a 2D axisymmetric eigenmode overlap calculation.

Model:
  [-∇^2 + U(rho,z;D)] psi = E psi
  U = m0^2 (Omega^2 - 1) + (1 - 6 xi) (lap_Omega / Omega)

For m=0 axisymmetry, we use a finite-volume-like symmetric generalized
eigenproblem:
  K psi = E M psi
where M carries the cylindrical rho-weight and enables stable eigensolve
with scipy.sparse.linalg.eigsh(..., M=M).

Then:
  V_eff = U + m0^2
  deltaV = V_eff - Vbar(r),  Vbar(r)=0.5 * ∫_{-1}^{1} dmu V_eff(r*sqrt(1-mu^2), r*mu)
  M12 = 2*pi * ∫ rho dr dz psi1 deltaV psi2
  chi = |M12| / sqrt(Gamma1 * Gamma2), with Gamma_l = omega^(4*l+5), l=1,2.

This script is intended to produce the first convergence table for D=12
and then extend to D=6/12/18 with identical numerical settings.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "output" / "chi_fp_2d"


@dataclass(frozen=True)
class PhysicalParams:
    a: float = 0.04
    eps: float = 0.1
    m0: float = 1.0
    xi: float = 0.14


@dataclass(frozen=True)
class Level:
    name: str
    dr: float
    dz: float


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


def build_generalized_operator(
    D: float,
    p: PhysicalParams,
    rho_max: float,
    z_max: float,
    dr: float,
    dz: float,
):
    nr = int(round(rho_max / dr))
    nz = int(round(2.0 * z_max / dz))
    if nr < 8 or nz < 8:
        raise ValueError("Grid too small for stable eigensolve.")

    rho = (np.arange(nr) + 0.5) * dr
    z = -z_max + (np.arange(nz) + 0.5) * dz
    rr, zz = np.meshgrid(rho, z, indexing="ij")

    uu = u_potential(rr, zz, D, p)

    # Radial weighted stiffness (symmetric).
    rho_ph = (np.arange(nr) + 1.0) * dr
    rho_mh = np.arange(nr) * dr
    main_r = (rho_ph + rho_mh) / (dr * dr)
    off_r = -rho_ph[:-1] / (dr * dr)
    k_r = diags([off_r, main_r, off_r], offsets=[-1, 0, 1], format="csr")

    # Axial stiffness.
    main_z = np.full(nz, 2.0 / (dz * dz))
    off_z = np.full(nz - 1, -1.0 / (dz * dz))
    t_z = diags([off_z, main_z, off_z], offsets=[-1, 0, 1], format="csr")

    r_diag = diags(rho, 0, format="csr")
    k_mat = kron(k_r, eye(nz, format="csr")) + kron(r_diag, t_z)
    k_mat = k_mat + diags((rho[:, None] * uu).ravel(), 0, format="csr")

    m_mat = kron(r_diag, eye(nz, format="csr"))
    return rho, z, rr, zz, uu, k_mat, m_mat


def normalize_modes(psi: np.ndarray, rho: np.ndarray, dr: float, dz: float) -> np.ndarray:
    w = 2.0 * math.pi * rho[:, None] * dr * dz
    out = psi.copy()
    for i in range(out.shape[2]):
        nrm = math.sqrt(np.sum(w * (out[:, :, i] ** 2)))
        out[:, :, i] /= nrm
    return out


def run_case(
    D: float,
    level: Level,
    p: PhysicalParams,
    rho_max: float,
    z_margin: float,
    n_mu: int,
    tol: float,
    maxiter: int,
    sigma: float | None,
):
    z_max = D / 2.0 + z_margin
    t0 = time.time()
    rho, z, rr, zz, uu, k_mat, m_mat = build_generalized_operator(
        D=D, p=p, rho_max=rho_max, z_max=z_max, dr=level.dr, dz=level.dz
    )
    t_build = time.time() - t0

    t1 = time.time()
    if sigma is None:
        evals, evecs = eigsh(
            k_mat,
            k=2,
            M=m_mat,
            which="SA",
            tol=tol,
            maxiter=maxiter,
        )
    else:
        evals, evecs = eigsh(
            k_mat,
            k=2,
            M=m_mat,
            sigma=sigma,
            which="LM",
            tol=tol,
            maxiter=maxiter,
        )
    t_solve = time.time() - t1

    idx = np.argsort(evals)
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    nr, nz = len(rho), len(z)
    psi = np.stack([evecs[:, 0].reshape(nr, nz), evecs[:, 1].reshape(nr, nz)], axis=2)
    psi = normalize_modes(psi, rho, level.dr, level.dz)

    psi1 = psi[:, :, 0]
    psi2 = psi[:, :, 1]
    w_int = 2.0 * math.pi * rho[:, None] * level.dr * level.dz
    overlap12 = float(np.sum(w_int * psi1 * psi2))

    vfull = uu + p.m0 * p.m0
    r_field = np.sqrt(rr * rr + zz * zz)
    r_line = np.linspace(0.0, float(r_field.max()), 1400)
    vbar_line = spherical_average_v_eff(r_line, D, p, n_mu=n_mu)
    vbar = np.interp(r_field.ravel(), r_line, vbar_line).reshape(r_field.shape)
    deltav = vfull - vbar

    m12 = float(2.0 * math.pi * np.sum((rho[:, None] * psi1 * deltav * psi2)) * level.dr * level.dz)

    e1, e2 = float(evals[0]), float(evals[1])
    omega1 = float(np.sqrt(max(e1 + p.m0 * p.m0, 0.0)))
    omega2 = float(np.sqrt(max(e2 + p.m0 * p.m0, 0.0)))

    # Eq. Gamma_{N,l} = omega * (omega M_*)^(4l+4), with M_*=1 and l=1,2.
    gamma1 = omega1 ** 9
    gamma2 = omega1 ** 13
    chi = float(abs(m12) / math.sqrt(gamma1 * gamma2)) if gamma1 > 0 and gamma2 > 0 else float("nan")

    norm1 = float(np.sum(w_int * psi1 * psi1))
    norm2 = float(np.sum(w_int * psi2 * psi2))

    return {
        "D": D,
        "level": level.name,
        "rho_max": rho_max,
        "z_max": z_max,
        "dr": level.dr,
        "dz": level.dz,
        "Nr": nr,
        "Nz": nz,
        "E1": e1,
        "E2": e2,
        "omega1": omega1,
        "omega2": omega2,
        "M12": m12,
        "chi": chi,
        "norm1": norm1,
        "norm2": norm2,
        "overlap12": overlap12,
        "build_s": t_build,
        "solve_s": t_solve,
    }


def main():
    ap = argparse.ArgumentParser(description="2D first-principles chi extraction (coarse/mid/fine).")
    ap.add_argument("--Ds", default="6,12,18", help="Comma-separated D values, e.g. 12 or 6,12,18")
    ap.add_argument("--rho-max", type=float, default=1.5)
    ap.add_argument("--z-margin", type=float, default=1.5, help="Use z_max = D/2 + z_margin")
    ap.add_argument("--n-mu", type=int, default=100, help="Gauss-Legendre points for spherical average")
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--maxiter", type=int, default=30000)
    ap.add_argument(
        "--sigma",
        type=float,
        default=2.55,
        help="Shift-invert target for eigsh. Use negative value to disable (SA mode).",
    )
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()

    d_list: List[float] = [float(s.strip()) for s in args.Ds.split(",") if s.strip()]
    levels = [
        Level("coarse", dr=0.10, dz=0.05),
        Level("mid", dr=0.08, dz=0.04),
        Level("fine", dr=0.06, dz=0.03),
    ]
    p = PhysicalParams()
    sigma = None if args.sigma < 0 else float(args.sigma)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dval in d_list:
        for lev in levels:
            print(f"[run] D={dval:g}, level={lev.name}, dr={lev.dr}, dz={lev.dz}")
            row = run_case(
                D=dval,
                level=lev,
                p=p,
                rho_max=args.rho_max,
                z_margin=args.z_margin,
                n_mu=args.n_mu,
                tol=args.tol,
                maxiter=args.maxiter,
                sigma=sigma,
            )
            rows.append(row)
            print(
                f"      E1={row['E1']:.6f}, E2={row['E2']:.6f}, "
                f"M12={row['M12']:.6e}, chi={row['chi']:.6e}, solve={row['solve_s']:.2f}s"
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["D", "level"]).reset_index(drop=True)

    # Relative errors vs fine per D.
    rel_rows = []
    for dval in sorted(df["D"].unique()):
        sub = df[df["D"] == dval].set_index("level")
        if "fine" not in sub.index:
            continue
        ref = sub.loc["fine"]
        for lev in ["coarse", "mid", "fine"]:
            if lev not in sub.index:
                continue
            cur = sub.loc[lev]
            rel_rows.append(
                {
                    "D": dval,
                    "level": lev,
                    "rel_E1_vs_fine": abs(cur["E1"] - ref["E1"]) / max(abs(ref["E1"]), 1e-30),
                    "rel_E2_vs_fine": abs(cur["E2"] - ref["E2"]) / max(abs(ref["E2"]), 1e-30),
                    "rel_M12_vs_fine": abs(cur["M12"] - ref["M12"]) / max(abs(ref["M12"]), 1e-30),
                    "rel_chi_vs_fine": abs(cur["chi"] - ref["chi"]) / max(abs(ref["chi"]), 1e-30),
                }
            )

    rel_df = pd.DataFrame(rel_rows).sort_values(["D", "level"]).reset_index(drop=True)

    out_main = outdir / f"chi_fp2d_D{'-'.join(str(int(d)) if float(d).is_integer() else str(d) for d in d_list)}.csv"
    out_rel = outdir / f"chi_fp2d_D{'-'.join(str(int(d)) if float(d).is_integer() else str(d) for d in d_list)}_relerr.csv"
    df.to_csv(out_main, index=False)
    rel_df.to_csv(out_rel, index=False)

    print("\n[done] wrote")
    print(f"  {out_main}")
    print(f"  {out_rel}")
    if len(rel_df) > 0:
        print("\n[summary] relative errors vs fine")
        print(rel_df.to_string(index=False))


if __name__ == "__main__":
    main()
