#!/usr/bin/env python3
"""
2D axisymmetric phase-space candidate extraction for g_N (preliminary).

This script evaluates a Weyl/WKB-style candidate on the same action-derived
potential used by the localized chi extraction:

  N_ps(E) = (1 / 4pi) * integral d^2x d^2p Theta(E - U - p^2)
          = 0.5 * integral rho dr dz [E - U(rho,z)]_+

for the axisymmetric configuration-space measure. For each D we compute
low-lying generalized eigenvalues lambda_{1,2,3}, then define

  g_raw,1 = 1
  g_raw,n = 1 + N_ps(lambda_n) - N_ps(lambda_1), n=2,3

and a normalized profile used for shape-comparison:

  ghat_n = g_raw,n / g_raw,3  (so ghat_3 = 1).

This is a candidate diagnostic only and is not used in the baseline scan.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import eigsh


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
    if nr < 12 or nz < 16:
        raise ValueError("Grid too small for stable eigensolve.")

    rho = (np.arange(nr) + 0.5) * dr
    z = -z_max + (np.arange(nz) + 0.5) * dz
    rr, zz = np.meshgrid(rho, z, indexing="ij")
    uu = u_potential(rr, zz, D, p)

    rho_ph = (np.arange(nr) + 1.0) * dr
    rho_mh = np.arange(nr) * dr
    main_r = (rho_ph + rho_mh) / (dr * dr)
    off_r = -rho_ph[:-1] / (dr * dr)
    k_r = diags([off_r, main_r, off_r], offsets=[-1, 0, 1], format="csr")

    main_z = np.full(nz, 2.0 / (dz * dz))
    off_z = np.full(nz - 1, -1.0 / (dz * dz))
    t_z = diags([off_z, main_z, off_z], offsets=[-1, 0, 1], format="csr")

    r_diag = diags(rho, 0, format="csr")
    k_mat = kron(k_r, eye(nz, format="csr")) + kron(r_diag, t_z)
    k_mat = k_mat + diags((rho[:, None] * uu).ravel(), 0, format="csr")
    m_mat = kron(r_diag, eye(nz, format="csr"))
    return rho, z, uu, k_mat, m_mat


def n_phase_space(E: float, U: np.ndarray, rho: np.ndarray, dr: float, dz: float) -> float:
    # N_ps(E) = 0.5 * integral rho dr dz [E-U]_+
    return float(0.5 * np.sum(rho[:, None] * np.maximum(E - U, 0.0)) * dr * dz)


def main() -> None:
    ap = argparse.ArgumentParser(description="2D phase-space candidate extraction for g_N profile.")
    ap.add_argument("--Ds", default="6,12,18")
    ap.add_argument("--rho-max", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    ap.add_argument("--outdir", default="/Users/boypatrick/codex/PSLT_20260202/output/gn_fp_2d")
    args = ap.parse_args()

    D_list = [float(s.strip()) for s in args.Ds.split(",") if s.strip()]
    levels = [
        Level("coarse", dr=0.12, dz=0.06),
        Level("mid", dr=0.08, dz=0.04),
        Level("fine", dr=0.06, dz=0.03),
    ]
    p = PhysicalParams()

    rows = []
    for D in D_list:
        for lev in levels:
            z_max = D / 2.0 + args.z_margin
            rho, z, uu, K, M = build_generalized_operator(D, p, args.rho_max, z_max, lev.dr, lev.dz)
            # Use smallest-algebraic generalized eigenvalues for branch-stable low modes.
            vals, _ = eigsh(K, k=6, M=M, which="SA", tol=args.tol, maxiter=args.maxiter)
            vals = np.sort(np.real(vals))[:3]
            Nvals = [n_phase_space(float(E), uu, rho, lev.dr, lev.dz) for E in vals]
            g_raw = [1.0 + max(nv - Nvals[0], 0.0) for nv in Nvals]
            ghat = [g / max(g_raw[2], 1e-30) for g in g_raw]

            rows.append(
                {
                    "D": D,
                    "level": lev.name,
                    "dr": lev.dr,
                    "dz": lev.dz,
                    "Nr": len(rho),
                    "Nz": len(z),
                    "lambda1": float(vals[0]),
                    "lambda2": float(vals[1]),
                    "lambda3": float(vals[2]),
                    "g1_raw": float(g_raw[0]),
                    "g2_raw": float(g_raw[1]),
                    "g3_raw": float(g_raw[2]),
                    "g1_hat": float(ghat[0]),
                    "g2_hat": float(ghat[1]),
                    "g3_hat": float(ghat[2]),
                }
            )

    df = pd.DataFrame(rows).sort_values(["D", "level"]).reset_index(drop=True)

    rel_rows = []
    for D in sorted(df["D"].unique()):
        sub = df[df["D"] == D].set_index("level")
        if "fine" not in sub.index:
            continue
        ref = sub.loc["fine"]
        for lev in ["coarse", "mid", "fine"]:
            cur = sub.loc[lev]
            rel_rows.append(
                {
                    "D": D,
                    "level": lev,
                    "rel_g1_hat_vs_fine": abs(cur["g1_hat"] - ref["g1_hat"]) / max(abs(ref["g1_hat"]), 1e-30),
                    "rel_g2_hat_vs_fine": abs(cur["g2_hat"] - ref["g2_hat"]) / max(abs(ref["g2_hat"]), 1e-30),
                    "rel_g3_hat_vs_fine": abs(cur["g3_hat"] - ref["g3_hat"]) / max(abs(ref["g3_hat"]), 1e-30),
                    "max_rel_ghat_vs_fine": max(
                        abs(cur["g1_hat"] - ref["g1_hat"]) / max(abs(ref["g1_hat"]), 1e-30),
                        abs(cur["g2_hat"] - ref["g2_hat"]) / max(abs(ref["g2_hat"]), 1e-30),
                        abs(cur["g3_hat"] - ref["g3_hat"]) / max(abs(ref["g3_hat"]), 1e-30),
                    ),
                }
            )
    rel_df = pd.DataFrame(rel_rows).sort_values(["D", "level"]).reset_index(drop=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(int(d)) for d in D_list)
    out_main = outdir / f"gn_phase_space_2d_D{tag}.csv"
    out_rel = outdir / f"gn_phase_space_2d_D{tag}_relerr.csv"
    df.to_csv(out_main, index=False)
    rel_df.to_csv(out_rel, index=False)

    print(df.to_string(index=False))
    print("\n[summary] rel errors vs fine")
    print(rel_df.to_string(index=False))
    print(f"\n[done] wrote {out_main}")
    print(f"[done] wrote {out_rel}")


if __name__ == "__main__":
    main()
