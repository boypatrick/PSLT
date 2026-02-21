#!/usr/bin/env python3
"""
Numerical checks for the geometric origin of the two-center conformal factor Ω.

Checks:
1) Normalization of the Plummer kernel
     rho_eps(r) = 3 eps^2 / [4 pi (r^2 + eps^2)^(5/2)]
   via 4*pi*int r^2 rho_eps dr = 1.

2) Radial Laplacian identity
     nabla^2 (1/sqrt(r^2+eps^2)) = -3 eps^2 / (r^2+eps^2)^(5/2)
   using a finite-difference radial operator.

Outputs:
  - output/omega_geom_origin/omega_geom_origin_checks.csv
  - output/omega_geom_origin/omega_geom_origin_samples.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    eps = 0.1

    outdir = ROOT / "output" / "omega_geom_origin"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Kernel normalization
    r_max = 200.0
    n_norm = 400000
    r_norm = np.linspace(0.0, r_max, n_norm)
    rho = 3.0 * eps**2 / (4.0 * np.pi * (r_norm**2 + eps**2) ** 2.5)
    norm_num = 4.0 * np.pi * np.trapezoid(r_norm**2 * rho, r_norm)

    # 2) Radial Laplacian identity
    r_min = 1.0e-4
    r_max_lap = 80.0
    n_lap = 500000
    r = np.linspace(r_min, r_max_lap, n_lap)
    f = 1.0 / np.sqrt(r**2 + eps**2)

    df = np.gradient(f, r, edge_order=2)
    lap_num = np.gradient((r**2) * df, r, edge_order=2) / (r**2)
    lap_exact = -3.0 * eps**2 / (r**2 + eps**2) ** 2.5

    abs_err = np.abs(lap_num - lap_exact)
    rel_err = abs_err / np.maximum(np.abs(lap_exact), 1e-300)

    # Exclude boundaries where finite differences are least accurate
    sl = slice(1000, -1000)
    # Report relative errors only in the dynamically relevant band where
    # |lap_exact| is not near floating-noise floor.
    band = (np.abs(lap_exact) > 1e-5)
    band_sl = np.zeros_like(band, dtype=bool)
    band_sl[sl] = True
    valid = band & band_sl

    max_abs = float(np.max(abs_err[sl]))
    max_rel = float(np.max(rel_err[valid]))
    med_rel = float(np.median(rel_err[valid]))

    checks = pd.DataFrame(
        [
            {
                "eps": eps,
                "norm_num": norm_num,
                "norm_abs_err": abs(norm_num - 1.0),
                "laplacian_max_abs_err": max_abs,
                "laplacian_max_rel_err": max_rel,
                "laplacian_median_rel_err": med_rel,
            }
        ]
    )
    checks_path = outdir / "omega_geom_origin_checks.csv"
    checks.to_csv(checks_path, index=False)

    sample_idx = np.linspace(1000, n_lap - 1001, 30, dtype=int)
    samples = pd.DataFrame(
        {
            "r": r[sample_idx],
            "lap_num": lap_num[sample_idx],
            "lap_exact": lap_exact[sample_idx],
            "abs_err": abs_err[sample_idx],
            "rel_err": rel_err[sample_idx],
        }
    )
    samples_path = outdir / "omega_geom_origin_samples.csv"
    samples.to_csv(samples_path, index=False)

    print(f"[saved] {checks_path}")
    print(f"[saved] {samples_path}")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
