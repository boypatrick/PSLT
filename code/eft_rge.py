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

from eft_wilson_matching import decompose_diag_offdiag


@dataclass(frozen=True)
class EFTFiniteOneLoopMatchConfig:
    kappa_diag: float = 0.0
    kappa_offdiag: float = 0.0
    mode: str = "constant"  # "constant" or "input_tied"
    input_diag_scale: float = 0.0
    input_offdiag_scale: float = 0.0
    floor: float = 1e-30


@dataclass(frozen=True)
class EFTLeadingLogRGEConfig:
    mu_low: float = 1.0
    gamma_diag: float = 2.0
    gamma_offdiag: float = 1.0
    log_clip: float = 6.0
    floor: float = 1e-30


@dataclass(frozen=True)
class FiniteOneLoopMatchWitness:
    c_tree_diag: np.ndarray
    c_tree_offdiag: np.ndarray
    delta_match_diag: np.ndarray
    delta_match_offdiag: np.ndarray
    c_match: np.ndarray
    kappa_diag: float
    kappa_offdiag: float
    kappa_diag_eff: float
    kappa_offdiag_eff: float
    finite_fac_diag: float
    finite_fac_offdiag: float
    mode: str
    shell_spread: float
    coeff_cv: float
    offdiag_mix: float


@dataclass(frozen=True)
class LeadingLogRGEWitness:
    c_match_diag: np.ndarray
    c_match_offdiag: np.ndarray
    delta_rge_diag: np.ndarray
    delta_rge_offdiag: np.ndarray
    c_low: np.ndarray
    mu_match: float
    mu_low: float
    log_ratio: float
    gamma_diag: float
    gamma_offdiag: float


def mu_match_from_m2(m2: np.ndarray, floor: float = 1e-30) -> float:
    """
    Effective UV matching scale from mediator masses M_N^2:
      mu_match = exp( mean_N log(sqrt(M_N^2)) ).
    """
    m2v = np.asarray(m2, dtype=float).reshape(-1)
    m = np.sqrt(np.maximum(np.abs(m2v), floor))
    return float(np.exp(np.mean(np.log(np.maximum(m, floor)))))


def finite_match_invariants(
    g_uv: np.ndarray,
    p_kin: np.ndarray,
    m2: np.ndarray,
    c_tree: np.ndarray,
    floor: float = 1e-30,
) -> dict[str, float]:
    """
    Dimensionless local invariants used for input-tied finite matching.

    - shell_spread: average absolute threshold-log around mu_match
    - coeff_cv: coefficient variation of P_N / M_N^2
    - offdiag_mix: Frobenius off-diagonal norm divided by diagonal trace
    """
    g = np.asarray(g_uv, dtype=float)
    if g.shape != (3, 3):
        raise ValueError(f"g_uv must have shape (3,3), got {g.shape}.")
    p = np.asarray(p_kin, dtype=float).reshape(3)
    p = np.maximum(p, 0.0)
    m2v = np.asarray(m2, dtype=float).reshape(3)
    m2v = np.maximum(np.abs(m2v), floor)
    c = np.asarray(c_tree, dtype=float)
    if c.shape != (3, 3):
        raise ValueError(f"c_tree must have shape (3,3), got {c.shape}.")

    mu_match = mu_match_from_m2(m2v, floor=floor)
    m = np.sqrt(np.maximum(m2v, floor))
    shell_logs = np.log(np.maximum(m, floor) / max(mu_match, floor))
    shell_spread = float(np.mean(np.abs(shell_logs)))

    coeff = p / m2v
    coeff_mean = float(np.mean(np.maximum(coeff, floor)))
    coeff_cv = float(np.std(coeff) / max(coeff_mean, floor))

    diag, off = decompose_diag_offdiag(c)
    diag_trace = float(np.trace(diag))
    offdiag_mix = float(np.linalg.norm(off, ord="fro") / max(diag_trace, floor))

    return {
        "mu_match": float(mu_match),
        "shell_spread": float(shell_spread),
        "coeff_cv": float(coeff_cv),
        "offdiag_mix": float(offdiag_mix),
    }


def resolve_finite_match_kappas(
    cfg: EFTFiniteOneLoopMatchConfig,
    g_uv: np.ndarray | None = None,
    p_kin: np.ndarray | None = None,
    m2: np.ndarray | None = None,
    c_tree: np.ndarray | None = None,
) -> dict[str, float]:
    mode = str(cfg.mode).strip().lower()
    if mode not in {"constant", "input_tied"}:
        raise ValueError(f"Unsupported finite-match mode '{cfg.mode}'.")

    shell_spread = 0.0
    coeff_cv = 0.0
    offdiag_mix = 0.0
    if mode == "input_tied":
        if g_uv is None or p_kin is None or m2 is None or c_tree is None:
            raise ValueError("input_tied finite matching requires g_uv, p_kin, m2, and c_tree.")
        inv = finite_match_invariants(g_uv=g_uv, p_kin=p_kin, m2=m2, c_tree=c_tree, floor=cfg.floor)
        shell_spread = float(inv["shell_spread"])
        coeff_cv = float(inv["coeff_cv"])
        offdiag_mix = float(inv["offdiag_mix"])

    kappa_diag_eff = float(cfg.kappa_diag)
    kappa_offdiag_eff = float(cfg.kappa_offdiag)
    if mode == "input_tied":
        kappa_diag_eff += float(cfg.input_diag_scale) * shell_spread * (1.0 + coeff_cv)
        kappa_offdiag_eff += float(cfg.input_offdiag_scale) * shell_spread * offdiag_mix

    return {
        "mode": mode,
        "kappa_diag_eff": float(kappa_diag_eff),
        "kappa_offdiag_eff": float(kappa_offdiag_eff),
        "shell_spread": float(shell_spread),
        "coeff_cv": float(coeff_cv),
        "offdiag_mix": float(offdiag_mix),
    }


def apply_ceh_finite_one_loop(
    c_tree: np.ndarray,
    cfg: EFTFiniteOneLoopMatchConfig,
    g_uv: np.ndarray | None = None,
    p_kin: np.ndarray | None = None,
    m2: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Minimal finite one-loop matching at mu_match:

      C_match = C_tree + (kappa / 16 pi^2) * C_tree

    with independent diagonal/off-diagonal coefficients.
    """
    c = np.asarray(c_tree, dtype=float)
    if c.shape != (3, 3):
        raise ValueError(f"c_tree must have shape (3,3), got {c.shape}.")

    resolved = resolve_finite_match_kappas(cfg=cfg, g_uv=g_uv, p_kin=p_kin, m2=m2, c_tree=c)

    loop = 16.0 * np.pi * np.pi
    fac_diag = 1.0 + float(resolved["kappa_diag_eff"]) / loop
    fac_off = 1.0 + float(resolved["kappa_offdiag_eff"]) / loop

    diag = np.diag(np.diag(c)) * fac_diag
    off = (c - np.diag(np.diag(c))) * fac_off
    c_match = np.maximum(diag + off, cfg.floor)

    return c_match, {
        "mode": str(resolved["mode"]),
        "kappa_diag": float(cfg.kappa_diag),
        "kappa_offdiag": float(cfg.kappa_offdiag),
        "kappa_diag_eff": float(resolved["kappa_diag_eff"]),
        "kappa_offdiag_eff": float(resolved["kappa_offdiag_eff"]),
        "shell_spread": float(resolved["shell_spread"]),
        "coeff_cv": float(resolved["coeff_cv"]),
        "offdiag_mix": float(resolved["offdiag_mix"]),
        "finite_fac_diag": float(fac_diag),
        "finite_fac_offdiag": float(fac_off),
    }


def finite_one_loop_witness(
    c_tree: np.ndarray,
    cfg: EFTFiniteOneLoopMatchConfig,
    g_uv: np.ndarray | None = None,
    p_kin: np.ndarray | None = None,
    m2: np.ndarray | None = None,
) -> FiniteOneLoopMatchWitness:
    c_match, meta = apply_ceh_finite_one_loop(c_tree=c_tree, cfg=cfg, g_uv=g_uv, p_kin=p_kin, m2=m2)
    c_tree_diag, c_tree_offdiag = decompose_diag_offdiag(c_tree)
    delta_diag, delta_offdiag = decompose_diag_offdiag(c_match - np.asarray(c_tree, dtype=float))
    return FiniteOneLoopMatchWitness(
        c_tree_diag=c_tree_diag,
        c_tree_offdiag=c_tree_offdiag,
        delta_match_diag=delta_diag,
        delta_match_offdiag=delta_offdiag,
        c_match=c_match,
        kappa_diag=float(meta["kappa_diag"]),
        kappa_offdiag=float(meta["kappa_offdiag"]),
        kappa_diag_eff=float(meta["kappa_diag_eff"]),
        kappa_offdiag_eff=float(meta["kappa_offdiag_eff"]),
        finite_fac_diag=float(meta["finite_fac_diag"]),
        finite_fac_offdiag=float(meta["finite_fac_offdiag"]),
        mode=str(meta["mode"]),
        shell_spread=float(meta["shell_spread"]),
        coeff_cv=float(meta["coeff_cv"]),
        offdiag_mix=float(meta["offdiag_mix"]),
    )


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


def leading_log_witness(
    c_match: np.ndarray,
    mu_match: float,
    cfg: EFTLeadingLogRGEConfig,
) -> LeadingLogRGEWitness:
    c_low, log_ratio = run_ceh_leading_log(c_match=c_match, mu_match=mu_match, cfg=cfg)
    c_match_diag, c_match_offdiag = decompose_diag_offdiag(c_match)
    delta_diag, delta_offdiag = decompose_diag_offdiag(c_low - np.asarray(c_match, dtype=float))
    return LeadingLogRGEWitness(
        c_match_diag=c_match_diag,
        c_match_offdiag=c_match_offdiag,
        delta_rge_diag=delta_diag,
        delta_rge_offdiag=delta_offdiag,
        c_low=c_low,
        mu_match=float(max(float(mu_match), cfg.floor)),
        mu_low=float(cfg.mu_low),
        log_ratio=float(log_ratio),
        gamma_diag=float(cfg.gamma_diag),
        gamma_offdiag=float(cfg.gamma_offdiag),
    )
