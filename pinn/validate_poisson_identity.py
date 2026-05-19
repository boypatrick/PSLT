#!/usr/bin/env python3
"""Finite-difference validation of the Plummer Poisson identity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_center_geometry import laplacian_plummer_analytic, plummer_kernel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--h", type=float, default=1.0e-4)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--radius-min", type=float, default=0.1)
    parser.add_argument("--radius-max", type=float, default=4.0)
    return parser.parse_args()


def finite_difference_laplacian(x: np.ndarray, y: np.ndarray, z: np.ndarray, eps: float, h: float) -> np.ndarray:
    f0 = plummer_kernel(x, y, z, eps)
    dxx = (plummer_kernel(x + h, y, z, eps) - 2.0 * f0 + plummer_kernel(x - h, y, z, eps)) / (h * h)
    dyy = (plummer_kernel(x, y + h, z, eps) - 2.0 * f0 + plummer_kernel(x, y - h, z, eps)) / (h * h)
    dzz = (plummer_kernel(x, y, z + h, eps) - 2.0 * f0 + plummer_kernel(x, y, z - h, eps)) / (h * h)
    return dxx + dyy + dzz


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    xyz = rng.normal(size=(args.n, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    radii = rng.uniform(args.radius_min, args.radius_max, size=(args.n, 1))
    pts = xyz * radii
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r2 = x * x + y * y + z * z

    fd = finite_difference_laplacian(x, y, z, args.eps, args.h)
    exact = laplacian_plummer_analytic(r2, args.eps)
    abs_err = np.abs(fd - exact)
    rel_err = abs_err / np.maximum(np.abs(exact), 1.0e-14)

    print("Plummer identity finite-difference check")
    print(f"eps={args.eps:g} h={args.h:g} n={args.n} seed={args.seed}")
    print(f"max_abs_error={abs_err.max():.6e}")
    print(f"median_abs_error={np.median(abs_err):.6e}")
    print(f"max_rel_error={rel_err.max():.6e}")
    print(f"median_rel_error={np.median(rel_err):.6e}")
    print("identity: laplacian[(r^2+eps^2)^(-1/2)] = -3 eps^2 (r^2+eps^2)^(-5/2)")


if __name__ == "__main__":
    main()

