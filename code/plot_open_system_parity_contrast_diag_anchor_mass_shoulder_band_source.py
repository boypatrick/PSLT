#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from audit_open_system_parity_contrast_self_mirror_source import (
    solve_fields,
    omega_single,
    spherical_average_generic,
)
from extract_chi_localized_2d import Level, PhysicalParams

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def _mass_field(rho: np.ndarray, z: np.ndarray, center: float, p: PhysicalParams) -> np.ndarray:
    om = omega_single(rho, z, center, p)
    return p.m0 * p.m0 * (om * om - 1.0)


def _local_negative_density(
    D: float,
    rho_anchor: float,
    zeta_cut: float,
    rho_max: float,
    z_margin: float,
    dr: float,
    dz: float,
    sigma: float,
    tol: float,
    maxiter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = PhysicalParams()
    level = Level("fine", dr=dr, dz=dz)
    f = solve_fields(
        D=D,
        p=p,
        level=level,
        rho_max=rho_max,
        z_margin=z_margin,
        sigma=sigma,
        tol=tol,
        maxiter=maxiter,
    )
    rr = f["rr"]
    zz = f["zz"]
    C = f["contrast_density"]
    r_field = f["r_field"]

    plus = (rr <= rho_anchor) & (np.abs(zz - D / 2.0) <= zeta_cut)
    minus = (rr <= rho_anchor) & (np.abs(zz + D / 2.0) <= zeta_cut)
    inner = plus | minus

    diag_density = np.zeros_like(C)
    diag_density[plus] += C[plus] * f["delta_plus"][plus]
    diag_density[minus] += C[minus] * f["delta_minus"][minus]
    sign_diag = 1.0 if float(np.sum(diag_density[inner])) >= 0.0 else -1.0

    r_line = np.linspace(0.0, float(r_field.max()), 1800)
    interp = lambda arr: np.interp(r_field.ravel(), r_line, arr).reshape(r_field.shape)
    up_mass = _mass_field(rr, zz, D / 2.0, p)
    um_mass = _mass_field(rr, zz, -D / 2.0, p)
    up_mass_bar = spherical_average_generic(
        r_line, D, p, lambda rho, z, D, p: _mass_field(rho, z, D / 2.0, p), 120
    )
    um_mass_bar = spherical_average_generic(
        r_line, D, p, lambda rho, z, D, p: _mass_field(rho, z, -D / 2.0, p), 120
    )
    delta_mass = np.zeros_like(C)
    delta_mass[plus] += C[plus] * (up_mass - interp(up_mass_bar))[plus]
    delta_mass[minus] += C[minus] * (um_mass - interp(um_mass_bar))[minus]
    q_mass = sign_diag * delta_mass
    q_mass_neg = np.clip(-q_mass, 0.0, None)

    zeta_p = np.abs(zz - D / 2.0)
    zeta_m = np.abs(zz + D / 2.0)
    local_zeta = np.where(plus, zeta_p, zeta_m)

    rho_vals = np.unique(rr[inner])
    zeta_vals = np.unique(local_zeta[inner])
    rho_idx = {float(v): i for i, v in enumerate(rho_vals)}
    zeta_idx = {float(v): i for i, v in enumerate(zeta_vals)}
    grid = np.zeros((len(zeta_vals), len(rho_vals)))
    for rho, zeta, val in zip(rr[inner].ravel(), local_zeta[inner].ravel(), q_mass_neg[inner].ravel()):
        grid[zeta_idx[float(zeta)], rho_idx[float(rho)]] += float(val)
    return rho_vals, zeta_vals, grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr-D", type=float, default=11.0)
    ap.add_argument("--rho-anchor", type=float, default=0.70)
    ap.add_argument("--zeta-cut", type=float, default=1.20)
    ap.add_argument("--rho-shoulder", type=float, default=0.20)
    ap.add_argument("--canonical-zeta-low", type=float, default=0.10)
    ap.add_argument("--rho-max-grid", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.06)
    ap.add_argument("--dz", type=float, default=0.03)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    args = ap.parse_args()

    detail = pd.read_csv(OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_shoulder_band_source_detail.csv")
    scan = pd.read_csv(OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_shoulder_band_source_scan.csv")

    zeta_vals_scan = np.sort(scan["zeta_low"].unique())
    core_vals = [float(scan.loc[np.isclose(scan["zeta_low"], z), "max_core_over_P_mass_inner"].iloc[0]) for z in zeta_vals_scan]
    share_vals = [float(scan.loc[np.isclose(scan["zeta_low"], z), "min_band_over_shoulder_share"].iloc[0]) for z in zeta_vals_scan]

    rho_local, zeta_local, neg_grid = _local_negative_density(
        D=float(args.repr_D),
        rho_anchor=float(args.rho_anchor),
        zeta_cut=float(args.zeta_cut),
        rho_max=float(args.rho_max_grid),
        z_margin=float(args.z_margin),
        dr=float(args.dr),
        dz=float(args.dz),
        sigma=float(args.sigma),
        tol=float(args.tol),
        maxiter=maxiter if (maxiter := int(args.maxiter)) else int(args.maxiter),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(zeta_vals_scan, core_vals, marker="o", label=r"$\max_D\,N_{\rm core}/P_{\rm mass,in}$")
    ax.plot(zeta_vals_scan, share_vals, marker="s", label=r"$\min_D\,N_{\rm band}/N_{\rm shoulder}$")
    ax.axvline(float(args.canonical_zeta_low), color="0.4", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\zeta_{\rm low}$")
    ax.set_ylabel("ratio")
    ax.set_title(r"Axial hollow scan inside $N_{\rm mass,shoulder}^{(0.20,1.20)}$")
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    ax.plot(detail["D"], detail["N_mass_shoulder"] / detail["P_mass_inner"], marker="o", label=r"$N_{\rm shoulder}/P_{\rm mass,in}$")
    ax.plot(detail["D"], detail["canonical_band_over_P_mass_inner"], marker="s", label=r"$N_{\rm band}/P_{\rm mass,in}$")
    ax.plot(detail["D"], detail["canonical_core_over_P_mass_inner"], marker="^", label=r"$N_{\rm core}/P_{\rm mass,in}$")
    ax.axvline(11.0, color="0.5", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$D$")
    ax.set_ylabel("ratio")
    ax.set_title(r"Canonical adverse band $(0.20;0.10,1.20)$")
    ax.legend(fontsize=8, loc="best")

    ax = axes[2]
    positive_vals = neg_grid[neg_grid > 0.0]
    vmin = float(max(positive_vals.min(), 1e-12)) if positive_vals.size else 1e-12
    vmax = float(max(neg_grid.max(), vmin * 10.0))
    im = ax.pcolormesh(
        rho_local,
        zeta_local,
        np.where(neg_grid > 0.0, neg_grid, np.nan),
        shading="nearest",
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.axvline(float(args.rho_shoulder), color="cyan", linestyle="--", linewidth=1.2)
    ax.axhline(float(args.canonical_zeta_low), color="cyan", linestyle="--", linewidth=1.2)
    ax.axhline(float(args.zeta_cut), color="white", linestyle=":", linewidth=1.0)
    ax.set_xlim(0.0, float(args.rho_anchor))
    ax.set_ylim(0.0, float(args.zeta_cut))
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\zeta=|z\mp D/2|$")
    ax.set_title(rf"Local $[q_{{\rm mass}}]_-$ at $D={args.repr_D:.0f}$")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$[q_{\rm mass}]_-$")

    out = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_shoulder_band_source_figure.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
