#!/usr/bin/env python3
"""
Tree-level EFT/Wilson matching utilities for PSLT H->ll observables.

This module currently supports two map-level matching closures:
  - projector-matched closure:
      y_raw(N), P_N^(kin), chi_eff(D) -> C_{eH}^{ij}
  - UV-inspired tree closure:
      g_{iN}(D), P_N^(kin), M_N^2(D) -> C_{eH}^{ij}
      with C = G diag(P_N/M_N^2) G^T.

Design notes:
  - In the strict diagonal limit (mix_scale=0), the projector-matched mode
    reduces to C_ii = y_raw_i * P_i when basis_mode="sqrt_yraw", reproducing
    the old diagonal ansatz at map level.
  - The UV-inspired tree closure removes the bounded nearest-neighbor flavor
    projector and uses overlap-extracted flavor-layer couplings directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EFTWilsonMatchConfig:
    basis_mode: str = "sqrt_yraw"  # "sqrt_yraw" or "yraw"
    mix_scale: float = 200.0
    mix_max: float = 0.25
    eta_power: float = 1.0
    eta_ref: float = 1.0
    width_mode: str = "sm_leptonic"  # "none" or "sm_leptonic"
    width_scale: float = 1.0
    br_ee: float = 5.0e-9
    br_mumu: float = 2.2e-4
    br_tautau: float = 6.3e-2
    floor: float = 1e-30


@dataclass(frozen=True)
class UVTreeMatchConfig:
    m2_floor: float = 1e-10
    coupling_floor: float = 1e-30


def mixing_epsilon(chi_eff: float, eta_val: float, cfg: EFTWilsonMatchConfig) -> float:
    eta_ratio = max(float(eta_val), cfg.floor) / max(float(cfg.eta_ref), cfg.floor)
    eps = float(cfg.mix_scale) * max(float(chi_eff), 0.0) * (eta_ratio ** float(cfg.eta_power))
    return float(np.clip(eps, 0.0, cfg.mix_max))


def flavor_projector(eps: float, floor: float = 1e-30) -> np.ndarray:
    """
    Build a bounded row-stochastic flavor-layer projector.
    Rows = flavors (e,mu,tau), cols = layers (N=1,2,3).
    """
    eps = float(np.clip(eps, 0.0, 0.49))
    proj = np.array(
        [
            [1.0 - eps, eps, 0.0],
            [0.5 * eps, 1.0 - eps, 0.5 * eps],
            [0.0, eps, 1.0 - eps],
        ],
        dtype=float,
    )
    proj = np.maximum(proj, floor)
    proj /= np.sum(proj, axis=1, keepdims=True)
    return proj


def flavor_layer_couplings(
    y_raw: np.ndarray,
    eps: float,
    cfg: EFTWilsonMatchConfig,
) -> np.ndarray:
    y = np.asarray(y_raw, dtype=float).reshape(3)
    y = np.maximum(y, cfg.floor)
    if cfg.basis_mode == "sqrt_yraw":
        base = np.sqrt(y)
    elif cfg.basis_mode == "yraw":
        base = y
    else:
        raise ValueError(f"Unsupported basis_mode='{cfg.basis_mode}'.")

    proj = flavor_projector(eps, floor=cfg.floor)
    # Row-wise flavor amplitudes with layer projection.
    return np.diag(base) @ proj


def wilson_matrix(
    y_raw: np.ndarray,
    p_kin: np.ndarray,
    eps: float,
    cfg: EFTWilsonMatchConfig,
) -> np.ndarray:
    y_mat = flavor_layer_couplings(y_raw=y_raw, eps=eps, cfg=cfg)
    p = np.asarray(p_kin, dtype=float).reshape(3)
    p = np.maximum(p, 0.0)
    return y_mat @ np.diag(p) @ y_mat.T


def wilson_matrix_uv_tree(
    g_uv: np.ndarray,
    p_kin: np.ndarray,
    m2: np.ndarray,
    cfg: UVTreeMatchConfig,
) -> np.ndarray:
    """
    UV-inspired tree-level closure:
      C_{eH}^{ij} = sum_N g_{iN} * (P_N^(kin) / M_N^2) * g_{jN}

    where:
      - g_uv is a 3x3 flavor-layer coupling matrix (rows=flavors, cols=layers),
      - p_kin is the 3-component kinetic occupancy vector,
      - m2 is the 3-component mediator-mass-squared vector.
    """
    g = np.asarray(g_uv, dtype=float)
    if g.shape != (3, 3):
        raise ValueError(f"g_uv must have shape (3,3), got {g.shape}.")

    p = np.asarray(p_kin, dtype=float).reshape(3)
    p = np.maximum(p, 0.0)
    m2v = np.asarray(m2, dtype=float).reshape(3)
    m2v = np.maximum(np.abs(m2v), cfg.m2_floor)

    g = np.maximum(g, cfg.coupling_floor)
    return g @ np.diag(p / m2v) @ g.T


def total_width_ratio(
    c_diag: np.ndarray,
    c_diag_ref: np.ndarray,
    cfg: EFTWilsonMatchConfig,
) -> float:
    if cfg.width_mode == "none":
        return 1.0
    if cfg.width_mode != "sm_leptonic":
        raise ValueError(f"Unsupported width_mode='{cfg.width_mode}'.")

    c = np.maximum(np.asarray(c_diag, dtype=float).reshape(3), cfg.floor)
    c_ref = np.maximum(np.asarray(c_diag_ref, dtype=float).reshape(3), cfg.floor)
    partial = (c / c_ref) ** 2

    br = np.array([cfg.br_ee, cfg.br_mumu, cfg.br_tautau], dtype=float)
    delta = float(np.sum(br * (partial - 1.0)))
    ratio = 1.0 + float(cfg.width_scale) * delta
    return float(max(ratio, cfg.floor))
