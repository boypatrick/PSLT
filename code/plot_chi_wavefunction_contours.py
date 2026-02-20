#!/usr/bin/env python3
"""
Plot 2D axisymmetric eigenmode contours for Appendix C.

Produces a 3x2 panel figure (D=6,12,18; columns: psi1, psi2) on fine grids.

Outputs:
  - output/chi_fp_2d/chi_wavefunction_contours_D6_12_18.png
  - paper/chi_wavefunction_contours_D6_12_18.png
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

from extract_chi_localized_2d import (
    PhysicalParams,
    build_generalized_operator,
    normalize_modes,
)


ROOT = Path("/Users/boypatrick/codex/PSLT_20260202")
OUTDIR = ROOT / "output" / "chi_fp_2d"
PAPER_DIR = ROOT / "paper"


def compute_modes_for_d(d: float, p: PhysicalParams) -> Dict[str, np.ndarray]:
    rho_max = 3.0
    z_max = d / 2.0 + 6.0
    dr, dz = 0.06, 0.03  # fine level in Appendix C

    rho, z, _, _, _, k_mat, m_mat = build_generalized_operator(
        D=d, p=p, rho_max=rho_max, z_max=z_max, dr=dr, dz=dz
    )

    evals, evecs = eigsh(
        k_mat,
        k=2,
        M=m_mat,
        sigma=2.5,
        which="LM",
        tol=1e-8,
        maxiter=30000,
    )
    idx = np.argsort(np.real(evals))
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])

    nr, nz = len(rho), len(z)
    psi = np.stack(
        [
            evecs[:, 0].reshape(nr, nz),
            evecs[:, 1].reshape(nr, nz),
        ],
        axis=2,
    )
    psi = normalize_modes(psi, rho, dr, dz)

    return {
        "D": np.array([d]),
        "rho": rho,
        "z": z,
        "lambda1": np.array([float(evals[0])]),
        "lambda2": np.array([float(evals[1])]),
        "psi1": psi[:, :, 0],
        "psi2": psi[:, :, 1],
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    d_values = [6.0, 12.0, 18.0]
    p = PhysicalParams(a=0.04, eps=0.1, m0=1.0, xi=0.14)
    records: List[Dict[str, np.ndarray]] = [compute_modes_for_d(d, p) for d in d_values]

    vmax = max(
        float(np.max(np.abs(rec["psi1"])))
        for rec in records
    )
    vmax = max(
        vmax,
        max(float(np.max(np.abs(rec["psi2"]))) for rec in records),
    )

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.0), constrained_layout=True)
    levels = np.linspace(-vmax, vmax, 41)

    for i, rec in enumerate(records):
        d = float(rec["D"][0])
        rho = rec["rho"]
        z = rec["z"]
        zz, rr = np.meshgrid(z, rho, indexing="xy")

        for j, key in enumerate(["psi1", "psi2"]):
            ax = axes[i, j]
            field = rec[key]
            cf = ax.contourf(zz, rr, field, levels=levels, cmap="RdBu_r", extend="both")
            ax.contour(zz, rr, field, levels=[0.0], colors="k", linewidths=0.7)
            if i == 0:
                mode_label = r"$\psi_1$ (even)" if key == "psi1" else r"$\psi_2$ (odd)"
                ax.set_title(mode_label, fontsize=13)
            ax.text(
                0.02,
                0.93,
                rf"$D={d:.0f},\ \lambda_1={float(rec['lambda1'][0]):.4f},\ \lambda_2={float(rec['lambda2'][0]):.4f}$",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )
            ax.set_xlim(z.min(), z.max())
            ax.set_ylim(0.0, rho.max())
            if i == 2:
                ax.set_xlabel(r"$z$")
            if j == 0:
                ax.set_ylabel(r"$\rho$")

    cbar = fig.colorbar(cf, ax=axes, orientation="vertical", fraction=0.022, pad=0.02)
    cbar.set_label(r"Mode amplitude $\psi(\rho,z)$")
    fig.suptitle(
        "Appendix C: 2D Axisymmetric Eigenmodes for Localized-Channel Extraction\n"
        r"Fine grids with $(d\rho,dz)=(0.06,0.03)$ for $D=\{6,12,18\}$",
        fontsize=15,
    )

    out_png = OUTDIR / "chi_wavefunction_contours_D6_12_18.png"
    paper_png = PAPER_DIR / "chi_wavefunction_contours_D6_12_18.png"
    fig.savefig(out_png, dpi=260)
    fig.savefig(paper_png, dpi=260)
    plt.close(fig)

    print(f"[done] wrote {out_png}")
    print(f"[done] wrote {paper_png}")


if __name__ == "__main__":
    main()
