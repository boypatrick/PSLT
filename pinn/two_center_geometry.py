#!/usr/bin/env python3
"""Analytic two-center conformal geometry utilities for PINN experiments."""

from __future__ import annotations

import numpy as np


def plummer_kernel(x: np.ndarray, y: np.ndarray, z: np.ndarray, eps: float) -> np.ndarray:
    """Return (x^2+y^2+z^2+eps^2)^(-1/2)."""
    return 1.0 / np.sqrt(x * x + y * y + z * z + eps * eps)


def plummer_density(r2: np.ndarray, eps: float) -> np.ndarray:
    """Normalized 3D Plummer density."""
    return 3.0 * eps * eps / (4.0 * np.pi * (r2 + eps * eps) ** 2.5)


def laplacian_plummer_analytic(r2: np.ndarray, eps: float) -> np.ndarray:
    """Analytic 3D Laplacian of the regulated Coulomb kernel."""
    return -3.0 * eps * eps / (r2 + eps * eps) ** 2.5


def omega_axisymmetric(rho: np.ndarray, z: np.ndarray, D: float, a: float, eps: float) -> np.ndarray:
    """Two-center conformal factor in axisymmetric coordinates."""
    r_plus = np.sqrt(rho * rho + (z - D / 2.0) ** 2 + eps * eps)
    r_minus = np.sqrt(rho * rho + (z + D / 2.0) ** 2 + eps * eps)
    return 1.0 + a * (1.0 / r_plus + 1.0 / r_minus)


def sigma_axisymmetric(rho: np.ndarray, z: np.ndarray, D: float, a: float, eps: float) -> np.ndarray:
    """Two-center smeared source sigma in axisymmetric coordinates."""
    r2_plus = rho * rho + (z - D / 2.0) ** 2
    r2_minus = rho * rho + (z + D / 2.0) ** 2
    return a * (plummer_density(r2_plus, eps) + plummer_density(r2_minus, eps))


def laplacian_omega_axisymmetric(rho: np.ndarray, z: np.ndarray, D: float, a: float, eps: float) -> np.ndarray:
    """Analytic Laplacian of Omega from the source identity."""
    return -4.0 * np.pi * sigma_axisymmetric(rho, z, D, a, eps)

