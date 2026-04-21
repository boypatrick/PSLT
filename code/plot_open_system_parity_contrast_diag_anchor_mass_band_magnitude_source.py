#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import pandas as pd

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


def _local_factor_fields(
    D: float,
    rho_anchor: float,
    zeta_cut: float,
    rho_band: float,
    zeta_band_low: float,
    rho_max: float,
    z_margin: float,
    dr: float,
    dz: float,
    sigma: float,
    tol: float,
    maxiter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    zeta_p = np.abs(zz - D / 2.0)
    zeta_m = np.abs(zz + D / 2.0)
    local_zeta = np.where(plus, zeta_p, zeta_m)
    band = inner & (rr >= rho_band) & (local_zeta >= zeta_band_low) & (local_zeta <= zeta_cut)

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

    local_mass = np.zeros_like(C)
    local_mass[plus] += (up_mass - interp(up_mass_bar))[plus]
    local_mass[minus] += (um_mass - interp(um_mass_bar))[minus]
    neg_carrier = np.clip(-(sign_diag * C), 0.0, None)

    rho_vals = np.unique(rr[band])
    zeta_vals = np.unique(local_zeta[band])
    rho_idx = {float(v): i for i, v in enumerate(rho_vals)}
    zeta_idx = {float(v): i for i, v in enumerate(zeta_vals)}
    kernel_grid = np.zeros((len(zeta_vals), len(rho_vals)))
    carrier_grid = np.zeros((len(zeta_vals), len(rho_vals)))
    product_grid = np.zeros((len(zeta_vals), len(rho_vals)))
    for rho, zeta, kval, cval in zip(rr[band].ravel(), local_zeta[band].ravel(), local_mass[band].ravel(), neg_carrier[band].ravel()):
        i = zeta_idx[float(zeta)]
        j = rho_idx[float(rho)]
        kernel_grid[i, j] += float(kval)
        carrier_grid[i, j] += float(cval)
        product_grid[i, j] += float(kval * cval)
    return rho_vals, zeta_vals, kernel_grid, product_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repr-D", type=float, default=11.0)
    ap.add_argument("--rho-anchor", type=float, default=0.70)
    ap.add_argument("--zeta-cut", type=float, default=1.20)
    ap.add_argument("--rho-band", type=float, default=0.20)
    ap.add_argument("--zeta-band-low", type=float, default=0.10)
    ap.add_argument("--rho-max-grid", type=float, default=3.0)
    ap.add_argument("--z-margin", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.06)
    ap.add_argument("--dz", type=float, default=0.03)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=30000)
    args = ap.parse_args()

    detail = pd.read_csv(OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_band_magnitude_source_detail.csv")
    rho_local, zeta_local, kernel_grid, product_grid = _local_factor_fields(
        D=float(args.repr_D),
        rho_anchor=float(args.rho_anchor),
        zeta_cut=float(args.zeta_cut),
        rho_band=float(args.rho_band),
        zeta_band_low=float(args.zeta_band_low),
        rho_max=float(args.rho_max_grid),
        z_margin=float(args.z_margin),
        dr=float(args.dr),
        dz=float(args.dz),
        sigma=float(args.sigma),
        tol=float(args.tol),
        maxiter=int(args.maxiter),
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(detail["D"], detail["kernel_min"], marker="o", label=r"$K_{\min}^{\rm band}$")
    ax.plot(detail["D"], detail["kernel_avg_weighted"], marker="s", label=r"$\bar K_{\rm band}^{(-)}$")
    ax.plot(detail["D"], detail["kernel_max"], marker="^", label=r"$K_{\max}^{\rm band}$")
    ax.set_xlabel(r"$D$")
    ax.set_ylabel("kernel value")
    ax.set_title(r"Positive one-center mass kernel on the canonical band")
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    ax.plot(detail["D"], detail["N_mass_band"], marker="o", label=r"$N_{\rm mass,band}$")
    ax.plot(detail["D"], detail["lower_product"], marker="s", label=r"$K_{\min}^{\rm band}W_{\rm band}^{(-)}$")
    ax.plot(detail["D"], detail["carrier_neg_moment"], marker="^", label=r"$W_{\rm band}^{(-)}$")
    ax.set_xlabel(r"$D$")
    ax.set_ylabel("magnitude")
    ax.set_title(r"Band magnitude vs carrier factorization")
    ax.legend(fontsize=8, loc="best")

    ax = axes[2]
    positive_vals = product_grid[product_grid > 0.0]
    vmin = float(max(positive_vals.min(), 1e-12)) if positive_vals.size else 1e-12
    vmax = float(max(product_grid.max(), vmin * 10.0))
    im = ax.pcolormesh(
        rho_local,
        zeta_local,
        np.where(product_grid > 0.0, product_grid, np.nan),
        shading="nearest",
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.axvline(float(args.rho_band), color="cyan", linestyle="--", linewidth=1.1)
    ax.axhline(float(args.zeta_band_low), color="cyan", linestyle="--", linewidth=1.1)
    ax.axhline(float(args.zeta_cut), color="white", linestyle=":", linewidth=1.0)
    ax.set_xlim(0.0, float(args.rho_anchor))
    ax.set_ylim(0.0, float(args.zeta_cut))
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\zeta=|z\mp D/2|$")
    ax.set_title(rf"Local band factor $K_{{\rm band}}W_{{\rm band}}^{{(-)}}$ at $D={args.repr_D:.0f}$")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$K_{\rm band} W_{\rm band}^{(-)}$")

    out = OUTDIR / "chi_open_system_parity_contrast_diag_anchor_mass_band_magnitude_source_figure.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
