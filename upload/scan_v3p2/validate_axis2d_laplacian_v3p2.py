#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_axis2d_laplacian_v3p2.py

TODO #2: 2D axisymmetric Laplacian validation in (rho,z) for the dual-center harmonic conformal factor.

We compute:
  Omega(rho,z) = 1 + a [1/sqrt(r_+^2+eps^2) + 1/sqrt(r_-^2+eps^2)]
  r_±^2 = rho^2 + (z ∓ D/2)^2

Analytic 3D Laplacian identity (valid in full 3D; in axisymmetric coords the same scalar Laplacian applies):
  ∇^2 (r^2+eps^2)^(-1/2) = -3 eps^2 (r^2+eps^2)^(-5/2)

Thus:
  ∇^2 Omega = a * Σ[-3 eps^2 (r_±^2+eps^2)^(-5/2)]

Derived potential (same as 1D scan engine):
  U(rho,z) = m0^2 (Omega^2 - 1) + (1-6 xi) (∇^2 Omega / Omega)

Deliverables:
  - U2D_{preset}_D{D}.npy (float32), grid metadata U2D_{preset}_grid_D{D}.npz
  - rel_error_axis_{preset}.png : compare U2D(rho=0,z) vs saved 1D U(z) from rep_axis1d_v3p1_*.csv
  - rel_error_axis_{preset}.csv : numeric error profile for each D

Default focuses on physical_gap preset.
"""

import argparse, os
import numpy as np
import pandas as pd

# ------------------ Presets (must match v3p1 engine) ------------------
PRESETS = {
    "diagnostic_deepwell": dict(a=8.0,   eps=0.25, m0=0.0, xi=0.0,  zmax=80.0),
    "physical_gap":        dict(a=0.040, eps=0.10, m0=1.0, xi=0.14, zmax=80.0),
}

def omega_rhoz(rho, z, D, a, eps):
    rp2 = rho**2 + (z - D/2.0)**2
    rm2 = rho**2 + (z + D/2.0)**2
    return 1.0 + a*(1.0/np.sqrt(rp2 + eps**2) + 1.0/np.sqrt(rm2 + eps**2))

def laplace_omega_rhoz(rho, z, D, a, eps):
    # Analytic 3D Laplacian of smoothed Coulomb kernel.
    rp2 = rho**2 + (z - D/2.0)**2
    rm2 = rho**2 + (z + D/2.0)**2
    return a*(-3.0*eps**2*(rp2 + eps**2)**(-2.5) + -3.0*eps**2*(rm2 + eps**2)**(-2.5))

def U_rhoz(rho, z, D, a, eps, m0, xi):
    Om = omega_rhoz(rho, z, D, a, eps)
    Lap = laplace_omega_rhoz(rho, z, D, a, eps)
    return m0**2*(Om**2 - 1.0) + (1.0 - 6.0*xi)*(Lap/Om)

def load_rep_U_1d(preset, D, base_dir):
    tag = "phys" if preset == "physical_gap" else "diag"
    fn = os.path.join(base_dir, f"rep_axis1d_v3p1_{tag}_U_D{D}.csv")
    df = pd.read_csv(fn)
    return df["z"].to_numpy(), df["U"].to_numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="physical_gap", choices=sorted(PRESETS.keys()))
    ap.add_argument("--Ds", default="6,12,18", help="comma-separated D list")
    ap.add_argument("--rho_max", type=float, default=20.0)
    ap.add_argument("--Nrho", type=int, default=501)
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    p = PRESETS[args.preset]
    a, eps, m0, xi, zmax = p["a"], p["eps"], p["m0"], p["xi"], p["zmax"]
    Ds = [float(s.strip()) for s in args.Ds.split(",") if s.strip()]

    # 2D grid
    rho = np.linspace(0.0, args.rho_max, args.Nrho)
    # use z grid from representative 1D file for D=first
    z1d, _ = load_rep_U_1d(args.preset, int(Ds[0]), args.out_dir)
    z = z1d.copy()

    # axis error profiles
    rows = []

    for D in Ds:
        Dint = int(D) if abs(D-int(D)) < 1e-12 else None

        # compute 2D U (analytic)
        R, Z = np.meshgrid(rho, z, indexing="ij")  # (Nrho, Nz)
        U2D = U_rhoz(R, Z, D, a, eps, m0, xi).astype(np.float32)

        # save
        out_npy = os.path.join(args.out_dir, f"U2D_{args.preset}_D{int(D)}.npy")
        np.save(out_npy, U2D)
        out_grid = os.path.join(args.out_dir, f"U2D_{args.preset}_grid_D{int(D)}.npz")
        np.savez(out_grid, rho=rho.astype(np.float32), z=z.astype(np.float32), D=float(D),
                 a=float(a), eps=float(eps), m0=float(m0), xi=float(xi))

        # compare axis with 1D saved U
        z1d, U1D = load_rep_U_1d(args.preset, int(D), args.out_dir)
        Uaxis2D = U2D[0, :].astype(np.float64)  # rho=0
        denom = np.maximum(np.max(np.abs(U1D)), 1e-15)
        rel = np.abs(Uaxis2D - U1D)/denom
        for zi, rerr in zip(z1d[::20], rel[::20]):  # downsample for csv size; full saved below separately
            rows.append(dict(preset=args.preset, D=float(D), z=float(zi), rel_error=float(rerr)))

        # save full axis error for this D
        df_full = pd.DataFrame({"z": z1d, "U1D": U1D, "U2D_axis": Uaxis2D, "rel_error": rel})
        df_full.to_csv(os.path.join(args.out_dir, f"rel_error_axis_{args.preset}_D{int(D)}.csv"), index=False)

    # combined downsampled table
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"rel_error_axis_{args.preset}_summary.csv"), index=False)

    # plot (max rel error per D)
    import matplotlib.pyplot as plt
    plt.figure()
    for D in Ds:
        df = pd.read_csv(os.path.join(args.out_dir, f"rel_error_axis_{args.preset}_D{int(D)}.csv"))
        plt.semilogy(df["z"], df["rel_error"], label=f"D={int(D)}")
    plt.xlabel("z")
    plt.ylabel("relative error |U2D(0,z)-U1D(z)| / max|U1D|")
    plt.title(f"Axis validation: {args.preset}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"rel_error_axis_{args.preset}.png"), dpi=200)

    print("Done. Outputs written to:", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()
