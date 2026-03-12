#!/usr/bin/env python3
"""
Minimal heat-kernel witnesses for PSLT conformal-background loop normalization.

This module does not attempt a full EYMH one-loop derivation. It provides a
scan-ready curved-background witness built from the scalar Laplace-type
operator in the conformal geometry:

  Delta = -nabla_g^2 + X,   X = m0^2 + xi R

and the local 4D Seeley-DeWitt coefficients up to a2:

  a0 = 1
  a1 = X + R/6
  a2 ~= 0.5 X^2 + (R X)/6 + R^2/72

The a2 term controls the logarithmic UV threshold in the local one-loop
effective action. We compress it into bounded diagonal/off-diagonal loop
normalization witnesses by averaging over well- and barrier-centered patches
with the covariant measure sqrt(g) d^3x ~ Omega^4 (2 pi rho d rho dz).

To isolate geometry-induced loop structure, we also provide vacuum-subtracted
(`*_geom`) coefficients where the flat-background contribution at Omega=1,
R=0 is removed:

  a1_flat = m0^2
  a2_flat = 0.5 * m0^4
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ConformalHeatKernelConfig:
    a: float = 0.04
    eps: float = 0.1
    m0: float = 1.0
    xi: float = 0.14
    rho_max: float = 3.0
    z_margin: float = 6.0
    dr: float = 0.15
    dz: float = 0.15
    sigma: float = 0.35
    floor: float = 1e-30


def omega_2center(rho: np.ndarray, z: np.ndarray, D: float, cfg: ConformalHeatKernelConfig) -> np.ndarray:
    rp2 = rho * rho + (z - D / 2.0) ** 2
    rm2 = rho * rho + (z + D / 2.0) ** 2
    return 1.0 + cfg.a * (1.0 / np.sqrt(rp2 + cfg.eps * cfg.eps) + 1.0 / np.sqrt(rm2 + cfg.eps * cfg.eps))


def lap_omega_2center(rho: np.ndarray, z: np.ndarray, D: float, cfg: ConformalHeatKernelConfig) -> np.ndarray:
    rp2 = rho * rho + (z - D / 2.0) ** 2
    rm2 = rho * rho + (z + D / 2.0) ** 2
    return cfg.a * (
        -3.0 * cfg.eps * cfg.eps
        * ((rp2 + cfg.eps * cfg.eps) ** (-2.5) + (rm2 + cfg.eps * cfg.eps) ** (-2.5))
    )


def ricci_scalar(omega: np.ndarray, lap_omega: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return -6.0 * lap_omega / np.maximum(omega, floor) ** 3


def scalar_potential_x(omega: np.ndarray, ricci: np.ndarray, cfg: ConformalHeatKernelConfig) -> np.ndarray:
    del omega  # retained for future higher-order extensions
    return np.full_like(ricci, cfg.m0 * cfg.m0) + cfg.xi * ricci


def seeley_dewitt_local(X: np.ndarray, R: np.ndarray) -> Dict[str, np.ndarray]:
    a0 = np.ones_like(R)
    a1 = X + R / 6.0
    # Minimal local a2 witness for a scalar Laplace-type operator.
    a2 = 0.5 * X * X + (R * X) / 6.0 + (R * R) / 72.0
    return {"a0": a0, "a1": a1, "a2": a2}


def seeley_dewitt_flat(cfg: ConformalHeatKernelConfig) -> Dict[str, float]:
    x_flat = cfg.m0 * cfg.m0
    return {
        "a0_flat": 1.0,
        "a1_flat": float(x_flat),
        "a2_flat": float(0.5 * x_flat * x_flat),
    }


def _patch_average(
    field: np.ndarray,
    rho: np.ndarray,
    z: np.ndarray,
    omega: np.ndarray,
    D: float,
    cfg: ConformalHeatKernelConfig,
    z0: float,
) -> float:
    rr, zz = np.meshgrid(rho, z, indexing="ij")
    sigma2 = max(cfg.sigma * cfg.sigma, cfg.floor)
    patch = np.exp(-(rr * rr + (zz - z0) * (zz - z0)) / (2.0 * sigma2))
    measure = 2.0 * math.pi * rr * cfg.dr * cfg.dz * np.maximum(omega, cfg.floor) ** 4
    weight = measure * patch
    denom = float(np.sum(weight))
    if denom <= cfg.floor:
        return 0.0
    return float(np.sum(weight * np.abs(field)) / denom)


def _grad_sq(field: np.ndarray, dr: float, dz: float) -> np.ndarray:
    fr = np.gradient(field, dr, axis=0, edge_order=2)
    fz = np.gradient(field, dz, axis=1, edge_order=2)
    return fr * fr + fz * fz


def conformal_heat_kernel_witness(D: float, cfg: ConformalHeatKernelConfig) -> Dict[str, float]:
    z_max = D / 2.0 + cfg.z_margin
    nr = max(12, int(round(cfg.rho_max / cfg.dr)))
    nz = max(24, int(round(2.0 * z_max / cfg.dz)))
    rho = (np.arange(nr) + 0.5) * cfg.dr
    z = -z_max + (np.arange(nz) + 0.5) * cfg.dz
    rr, zz = np.meshgrid(rho, z, indexing="ij")

    Om = omega_2center(rr, zz, D, cfg)
    lap_Om = lap_omega_2center(rr, zz, D, cfg)
    R = ricci_scalar(Om, lap_Om, floor=cfg.floor)
    X = scalar_potential_x(Om, R, cfg)
    R_abs = np.abs(R)
    X_geom_abs = np.abs(X - cfg.m0 * cfg.m0)
    grad_X_geom = _grad_sq(X_geom_abs, cfg.dr, cfg.dz)
    sdw = seeley_dewitt_local(X, R)
    flat = seeley_dewitt_flat(cfg)
    a1_geom = sdw["a1"] - flat["a1_flat"]
    a2_geom = sdw["a2"] - flat["a2_flat"]

    left_R_abs = _patch_average(R_abs, rho, z, Om, D, cfg, -D / 2.0)
    right_R_abs = _patch_average(R_abs, rho, z, Om, D, cfg, +D / 2.0)
    barrier_R_abs = _patch_average(R_abs, rho, z, Om, D, cfg, 0.0)
    left_X_abs = _patch_average(X_geom_abs, rho, z, Om, D, cfg, -D / 2.0)
    right_X_abs = _patch_average(X_geom_abs, rho, z, Om, D, cfg, +D / 2.0)
    barrier_X_abs = _patch_average(X_geom_abs, rho, z, Om, D, cfg, 0.0)
    barrier_grad_X = _patch_average(grad_X_geom, rho, z, Om, D, cfg, 0.0)

    left_a1 = _patch_average(sdw["a1"], rho, z, Om, D, cfg, -D / 2.0)
    right_a1 = _patch_average(sdw["a1"], rho, z, Om, D, cfg, +D / 2.0)
    left_a2 = _patch_average(sdw["a2"], rho, z, Om, D, cfg, -D / 2.0)
    right_a2 = _patch_average(sdw["a2"], rho, z, Om, D, cfg, +D / 2.0)
    barrier_a2 = _patch_average(sdw["a2"], rho, z, Om, D, cfg, 0.0)
    left_a1_geom = _patch_average(a1_geom, rho, z, Om, D, cfg, -D / 2.0)
    right_a1_geom = _patch_average(a1_geom, rho, z, Om, D, cfg, +D / 2.0)
    left_a2_geom = _patch_average(a2_geom, rho, z, Om, D, cfg, -D / 2.0)
    right_a2_geom = _patch_average(a2_geom, rho, z, Om, D, cfg, +D / 2.0)
    barrier_a2_geom = _patch_average(a2_geom, rho, z, Om, D, cfg, 0.0)

    well_a1 = 0.5 * (left_a1 + right_a1)
    well_a2 = 0.5 * (left_a2 + right_a2)
    well_a1_geom = 0.5 * (left_a1_geom + right_a1_geom)
    well_a2_geom = 0.5 * (left_a2_geom + right_a2_geom)
    hk_diag_density = float(well_a2)
    hk_barrier_density = float(barrier_a2)
    hk_abs_diag = float(hk_diag_density / (1.0 + hk_diag_density))
    hk_abs_offdiag = float(hk_barrier_density / (1.0 + hk_barrier_density))
    hk_barrier_ratio = float(hk_barrier_density / max(hk_diag_density, cfg.floor))
    hk_diag_density_geom = float(well_a2_geom)
    hk_barrier_density_geom = float(barrier_a2_geom)
    hk_abs_diag_geom = float(hk_diag_density_geom / (1.0 + hk_diag_density_geom))
    hk_abs_offdiag_geom = float(hk_barrier_density_geom / (1.0 + hk_barrier_density_geom))
    hk_barrier_ratio_geom = float(hk_barrier_density_geom / max(hk_diag_density_geom, cfg.floor))
    well_R_abs = 0.5 * (left_R_abs + right_R_abs)
    well_X_abs = 0.5 * (left_X_abs + right_X_abs)
    hk_curv_contrast_log = float(np.log10(max(well_R_abs, cfg.floor) / max(barrier_R_abs, cfg.floor)))
    hk_curv_access = float(1.0 / np.sqrt(1.0 + max(hk_curv_contrast_log, 0.0)))
    hk_barrier_stiffness_log = float(np.log10(1.0 + 1.0e6 * max(barrier_grad_X, 0.0)))
    hk_loop_prefactor_diag = float(
        hk_curv_access * np.sqrt(max(hk_abs_diag_geom, cfg.floor))
    )
    hk_loop_prefactor_offdiag = float(
        hk_barrier_ratio_geom / max(1.0 + hk_barrier_stiffness_log, cfg.floor)
    )
    hk_mass_access_diag = float(
        np.sqrt(max(well_X_abs, cfg.floor) / (1.0 + max(well_X_abs, cfg.floor)))
    )
    hk_mass_access_offdiag = float(
        np.sqrt(max(barrier_X_abs, cfg.floor) / (1.0 + max(barrier_X_abs, cfg.floor)))
    )
    hk_curv_screen_diag = float(1.0 / np.sqrt(1.0 + max(barrier_R_abs, 0.0)))
    hk_curv_screen_offdiag = float(
        hk_curv_access / max(1.0 + hk_barrier_stiffness_log, cfg.floor)
    )
    hk_loop_local_prefactor_diag = float(
        hk_loop_prefactor_diag * hk_mass_access_diag * hk_curv_screen_diag
    )
    hk_loop_local_prefactor_offdiag = float(
        hk_loop_prefactor_offdiag * hk_mass_access_offdiag * hk_curv_screen_offdiag
    )

    iz0 = int(np.argmin(np.abs(z)))
    ir0 = 0
    return {
        "hk_omega_mid": float(Om[ir0, iz0]),
        "hk_R_mid": float(R[ir0, iz0]),
        "hk_X_mid": float(X[ir0, iz0]),
        "hk_a1_well": float(well_a1),
        "hk_a2_well": float(well_a2),
        "hk_a2_barrier": float(barrier_a2),
        "hk_diag_density": float(hk_diag_density),
        "hk_barrier_density": float(hk_barrier_density),
        "hk_abs_diag": float(hk_abs_diag),
        "hk_abs_offdiag": float(hk_abs_offdiag),
        "hk_barrier_ratio": float(hk_barrier_ratio),
        "hk_a1_flat": float(flat["a1_flat"]),
        "hk_a2_flat": float(flat["a2_flat"]),
        "hk_a1_well_geom": float(well_a1_geom),
        "hk_a2_well_geom": float(well_a2_geom),
        "hk_a2_barrier_geom": float(barrier_a2_geom),
        "hk_diag_density_geom": float(hk_diag_density_geom),
        "hk_barrier_density_geom": float(hk_barrier_density_geom),
        "hk_abs_diag_geom": float(hk_abs_diag_geom),
        "hk_abs_offdiag_geom": float(hk_abs_offdiag_geom),
        "hk_barrier_ratio_geom": float(hk_barrier_ratio_geom),
        "hk_R_well_abs": float(well_R_abs),
        "hk_R_barrier_abs": float(barrier_R_abs),
        "hk_X_well_abs": float(well_X_abs),
        "hk_X_barrier_abs": float(barrier_X_abs),
        "hk_gradX_barrier": float(barrier_grad_X),
        "hk_curv_contrast_log": float(hk_curv_contrast_log),
        "hk_curv_access": float(hk_curv_access),
        "hk_barrier_stiffness_log": float(hk_barrier_stiffness_log),
        "hk_mass_access_diag": float(hk_mass_access_diag),
        "hk_mass_access_offdiag": float(hk_mass_access_offdiag),
        "hk_curv_screen_diag": float(hk_curv_screen_diag),
        "hk_curv_screen_offdiag": float(hk_curv_screen_offdiag),
        "hk_loop_prefactor_diag": float(hk_loop_prefactor_diag),
        "hk_loop_prefactor_offdiag": float(hk_loop_prefactor_offdiag),
        "hk_loop_local_prefactor_diag": float(hk_loop_local_prefactor_diag),
        "hk_loop_local_prefactor_offdiag": float(hk_loop_local_prefactor_offdiag),
    }
