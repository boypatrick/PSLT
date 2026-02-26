#!/usr/bin/env python3
"""
Leading-log RGE utilities for PSLT UV-tree Wilson matching.

This module provides a minimal scan-ready running step from a UV matching
scale mu_match to a fixed low scale mu_low:

  C(mu_low) = C(mu_match) + (gamma / 16 pi^2) * ln(mu_match/mu_low) * C(mu_match)

with separate anomalous-dimension proxies for diagonal and off-diagonal blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EFTFiniteOneLoopMatchConfig:
    kappa_diag: float = 0.0
    kappa_offdiag: float = 0.0
    floor: float = 1e-30


@dataclass(frozen=True)
class EFTLeadingLogRGEConfig:
    mu_low: float = 1.0
    gamma_diag: float = 2.0
    gamma_offdiag: float = 1.0
    log_clip: float = 6.0
    floor: float = 1e-30


def mu_match_from_m2(m2: np.ndarray, floor: float = 1e-30) -> float:
    """
    Effective UV matching scale from mediator masses M_N^2:
      mu_match = exp( mean_N log(sqrt(M_N^2)) ).
    """
    m2v = np.asarray(m2, dtype=float).reshape(-1)
    m = np.sqrt(np.maximum(np.abs(m2v), floor))
    return float(np.exp(np.mean(np.log(np.maximum(m, floor)))))


def apply_ceh_finite_one_loop(
    c_tree: np.ndarray,
    cfg: EFTFiniteOneLoopMatchConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Minimal finite one-loop matching at mu_match:

      C_match = C_tree + (kappa / 16 pi^2) * C_tree

    with independent diagonal/off-diagonal coefficients.
    """
    c = np.asarray(c_tree, dtype=float)
    if c.shape != (3, 3):
        raise ValueError(f"c_tree must have shape (3,3), got {c.shape}.")

    loop = 16.0 * np.pi * np.pi
    fac_diag = 1.0 + float(cfg.kappa_diag) / loop
    fac_off = 1.0 + float(cfg.kappa_offdiag) / loop

    diag = np.diag(np.diag(c)) * fac_diag
    off = (c - np.diag(np.diag(c))) * fac_off
    c_match = np.maximum(diag + off, cfg.floor)

    return c_match, {
        "kappa_diag": float(cfg.kappa_diag),
        "kappa_offdiag": float(cfg.kappa_offdiag),
        "finite_fac_diag": float(fac_diag),
        "finite_fac_offdiag": float(fac_off),
    }


def run_ceh_leading_log(
    c_match: np.ndarray,
    mu_match: float,
    cfg: EFTLeadingLogRGEConfig,
) -> tuple[np.ndarray, float]:
    """
    Apply leading-log running with blockwise anomalous-dimension proxies.

    Returns:
      (C_low, log_ratio)
    """
    c = np.asarray(c_match, dtype=float)
    if c.shape != (3, 3):
        raise ValueError(f"c_match must have shape (3,3), got {c.shape}.")

    mu_u = max(float(mu_match), cfg.floor)
    mu_l = max(float(cfg.mu_low), cfg.floor)
    log_ratio = float(np.log(mu_u / mu_l))
    log_ratio = float(np.clip(log_ratio, -cfg.log_clip, cfg.log_clip))

    loop = 16.0 * np.pi * np.pi
    fac_diag = 1.0 + float(cfg.gamma_diag) * log_ratio / loop
    fac_off = 1.0 + float(cfg.gamma_offdiag) * log_ratio / loop

    diag = np.diag(np.diag(c)) * fac_diag
    off = (c - np.diag(np.diag(c))) * fac_off
    c_low = diag + off
    return np.maximum(c_low, cfg.floor), log_ratio
