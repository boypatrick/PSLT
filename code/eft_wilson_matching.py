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
    width_mode: str = "sm_leptonic"  # "none", "sm_leptonic", "sm_leptonic_aniso_power", "sm_leptonic_aniso_band_power", or "sm_leptonic_aniso_band_reboost"
    width_scale: float = 1.0
    width_power_base: float = 1.0
    width_power_log_ratio_coeff: float = 0.0
    width_power_min: float = 1.0
    width_power_max: float = 1.0
    width_power_trigger_lo: float = 1.0
    width_power_trigger_hi: float = 1.0
    width_power_turnoff_lo: float = 1.0
    width_power_turnoff_hi: float = 1.0
    width_power_tail_logratio_lo: float = 0.0
    width_power_tail_logratio_hi: float = 0.0
    width_power_tail_reboost_max: float = 0.0
    br_ee: float = 5.0e-9
    br_mumu: float = 2.2e-4
    br_tautau: float = 6.3e-2
    floor: float = 1e-30


@dataclass(frozen=True)
class UVTreeMatchConfig:
    m2_floor: float = 1e-10
    coupling_floor: float = 1e-30


@dataclass(frozen=True)
class UVTreeOperatorBasisWitness:
    g_uv: np.ndarray
    p_kin: np.ndarray
    m2: np.ndarray
    coefficients: np.ndarray
    basis_matrices: np.ndarray
    c_tree: np.ndarray


def decompose_diag_offdiag(cmat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(cmat, dtype=float)
    if c.shape != (3, 3):
        raise ValueError(f"Matrix must have shape (3,3), got {c.shape}.")
    diag = np.diag(np.diag(c))
    off = c - diag
    return diag, off


def reconstruct_from_layer_basis(coefficients: np.ndarray, basis_matrices: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=float).reshape(3)
    basis = np.asarray(basis_matrices, dtype=float)
    if basis.shape != (3, 3, 3):
        raise ValueError(f"basis_matrices must have shape (3,3,3), got {basis.shape}.")
    c = np.zeros((3, 3), dtype=float)
    for n in range(3):
        c += float(coeff[n]) * basis[:, :, n]
    return c


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


def uv_tree_operator_basis(
    g_uv: np.ndarray,
    p_kin: np.ndarray,
    m2: np.ndarray,
    cfg: UVTreeMatchConfig,
) -> UVTreeOperatorBasisWitness:
    """
    Explicit layer-resolved operator-basis witness for the UV-tree closure:

      C_{eH}^{tree} = sum_N c_N * B_N
      c_N = P_N^(kin) / M_N^2
      B_N = g_N g_N^T

    where g_N is the N-th column of the overlap-extracted flavor-layer matrix.
    """
    g = np.asarray(g_uv, dtype=float)
    if g.shape != (3, 3):
        raise ValueError(f"g_uv must have shape (3,3), got {g.shape}.")

    p = np.asarray(p_kin, dtype=float).reshape(3)
    p = np.maximum(p, 0.0)
    m2v = np.asarray(m2, dtype=float).reshape(3)
    m2v = np.maximum(np.abs(m2v), cfg.m2_floor)
    g = np.maximum(g, cfg.coupling_floor)

    coeff = p / m2v
    basis = np.zeros((3, 3, 3), dtype=float)
    for n in range(3):
        col = g[:, n]
        basis[:, :, n] = np.outer(col, col)

    c_tree = reconstruct_from_layer_basis(coefficients=coeff, basis_matrices=basis)
    return UVTreeOperatorBasisWitness(
        g_uv=g,
        p_kin=p,
        m2=m2v,
        coefficients=coeff,
        basis_matrices=basis,
        c_tree=c_tree,
    )


def total_width_ratio(
    c_diag: np.ndarray,
    c_diag_ref: np.ndarray,
    cfg: EFTWilsonMatchConfig,
) -> float:
    if cfg.width_mode == "none":
        return 1.0
    if cfg.width_mode not in {
        "sm_leptonic",
        "sm_leptonic_aniso_power",
        "sm_leptonic_aniso_band_power",
        "sm_leptonic_aniso_band_reboost",
    }:
        raise ValueError(f"Unsupported width_mode='{cfg.width_mode}'.")

    c = np.maximum(np.asarray(c_diag, dtype=float).reshape(3), cfg.floor)
    c_ref = np.maximum(np.asarray(c_diag_ref, dtype=float).reshape(3), cfg.floor)
    partial = (c / c_ref) ** 2
    br = np.array([cfg.br_ee, cfg.br_mumu, cfg.br_tautau], dtype=float)
    if cfg.width_mode in {
        "sm_leptonic_aniso_power",
        "sm_leptonic_aniso_band_power",
        "sm_leptonic_aniso_band_reboost",
    }:
        raw_delta = float(np.sum(br * (partial - 1.0)))
        raw_ratio = 1.0 + float(cfg.width_scale) * raw_delta
        log_partial_ratio = float(np.log(max(float(partial[1]), cfg.floor) / max(float(partial[2]), cfg.floor)))
        rho_target = float(cfg.width_power_base) + float(cfg.width_power_log_ratio_coeff) * log_partial_ratio
        rho_target = float(np.clip(rho_target, float(cfg.width_power_min), float(cfg.width_power_max)))
        lo = max(float(cfg.width_power_trigger_lo), 1.0 + cfg.floor)
        hi = max(float(cfg.width_power_trigger_hi), lo + cfg.floor)
        if hi <= lo * (1.0 + 1e-12):
            activation = 1.0 if raw_ratio > lo else 0.0
        else:
            activation = float(
                np.clip(
                    np.log(max(raw_ratio, 1.0 + cfg.floor) / lo) / np.log(hi / lo),
                    0.0,
                    1.0,
                )
            )
        activation_raw = activation
        if cfg.width_mode in {"sm_leptonic_aniso_band_power", "sm_leptonic_aniso_band_reboost"}:
            turnoff_lo = max(float(cfg.width_power_turnoff_lo), hi)
            turnoff_hi = max(float(cfg.width_power_turnoff_hi), turnoff_lo + cfg.floor)
            if raw_ratio <= turnoff_lo:
                suppression = 1.0
            elif turnoff_hi <= turnoff_lo * (1.0 + 1e-12):
                suppression = 0.0 if raw_ratio > turnoff_lo else 1.0
            else:
                suppression = 1.0 - float(
                    np.clip(
                        np.log(max(raw_ratio, turnoff_lo) / turnoff_lo) / np.log(turnoff_hi / turnoff_lo),
                        0.0,
                        1.0,
                    )
                )
            activation *= suppression
            if cfg.width_mode == "sm_leptonic_aniso_band_reboost":
                tail_lo = max(float(cfg.width_power_tail_logratio_lo), 0.0)
                tail_hi = max(float(cfg.width_power_tail_logratio_hi), tail_lo + cfg.floor)
                log_ratio_mag = abs(log_partial_ratio)
                if log_ratio_mag <= tail_lo:
                    tail_reboost = 0.0
                elif tail_hi <= tail_lo + cfg.floor:
                    tail_reboost = float(cfg.width_power_tail_reboost_max)
                else:
                    tail_reboost = float(cfg.width_power_tail_reboost_max) * float(
                        np.clip((log_ratio_mag - tail_lo) / (tail_hi - tail_lo), 0.0, 1.0)
                    )
                activation += activation_raw * (1.0 - suppression) * tail_reboost
        rho = 1.0 - activation * (1.0 - rho_target)
        partial = np.maximum(partial, cfg.floor) ** rho

    delta = float(np.sum(br * (partial - 1.0)))
    ratio = 1.0 + float(cfg.width_scale) * delta
    return float(max(ratio, cfg.floor))
