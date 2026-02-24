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
