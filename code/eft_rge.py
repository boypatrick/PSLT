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
    mode: str = "constant"  # "constant", "input_tied", "action_normalized", or "action_absolute"
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
    gap_cv: float
    gap_asym: float
    g_col_norm_cv: float
    c_tree_diag_cv: float
    pkin_entropy: float
    action_norm_diag: float
    action_norm_offdiag: float
    coeff_l1: float
    coeff_l2: float
    coeff_align: float
    action_abs_diag: float
    action_abs_offdiag: float


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


def parent_action_invariants(
    g_uv: np.ndarray,
    p_kin: np.ndarray,
    m2: np.ndarray,
    c_tree: np.ndarray,
    floor: float = 1e-30,
) -> dict[str, float]:
    """
    Parent-action-side invariants for structured finite-threshold normalization.

    These are intended to be action-side witnesses rather than a completed loop
    derivation. They summarize:
      - shell-gap dispersion in sqrt(M_N^2),
      - layer-coupling norm dispersion in g_uv,
      - diagonal block dispersion in C_tree,
      - kinetic occupancy spread through normalized Shannon entropy,
      - coefficient-alignment witnesses for absolute normalization.
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

    shell = np.sort(np.sqrt(np.maximum(m2v, floor)))
    gaps = np.diff(shell)
    gap_mean = float(np.mean(np.maximum(gaps, floor))) if gaps.size else 0.0
    gap_cv = float(np.std(gaps) / max(gap_mean, floor)) if gaps.size else 0.0
    gap_asym = float(np.max(gaps) / max(np.min(np.maximum(gaps, floor)), floor) - 1.0) if gaps.size else 0.0

    g_col_norm = np.linalg.norm(g, axis=0)
    g_col_mean = float(np.mean(np.maximum(g_col_norm, floor)))
    g_col_norm_cv = float(np.std(g_col_norm) / max(g_col_mean, floor))

    c_tree_diag = np.diag(c)
    c_tree_diag_mean = float(np.mean(np.maximum(c_tree_diag, floor)))
    c_tree_diag_cv = float(np.std(c_tree_diag) / max(c_tree_diag_mean, floor))

    p_norm = p / max(float(np.sum(p)), floor)
    pkin_entropy = float(-np.sum(p_norm * np.log(np.maximum(p_norm, floor))) / np.log(float(len(p_norm))))

    coeff = p / m2v
    coeff_l1 = float(np.sum(np.abs(coeff)))
    coeff_l2 = float(np.linalg.norm(coeff))
    coeff_align = float(coeff_l2 / max(coeff_l1, floor))

    # Structured normalization witnesses kept O(1) by construction.
    action_norm_diag = float(
        1.0
        + 0.5 * gap_cv
        + 0.25 * g_col_norm_cv
        + 0.25 * c_tree_diag_cv
    )
    action_norm_offdiag = float(
        1.0
        + 0.5 * gap_cv
        + 0.5 * gap_asym / max(1.0 + gap_asym, 1.0)
    )
    action_abs_diag = float(pkin_entropy * coeff_align)
    action_abs_offdiag = float(action_abs_diag * gap_cv / max(1.0 + gap_asym, 1.0))

    return {
        "gap_cv": float(gap_cv),
        "gap_asym": float(gap_asym),
        "g_col_norm_cv": float(g_col_norm_cv),
        "c_tree_diag_cv": float(c_tree_diag_cv),
        "pkin_entropy": float(pkin_entropy),
        "action_norm_diag": float(action_norm_diag),
        "action_norm_offdiag": float(action_norm_offdiag),
        "coeff_l1": float(coeff_l1),
        "coeff_l2": float(coeff_l2),
        "coeff_align": float(coeff_align),
        "action_abs_diag": float(action_abs_diag),
        "action_abs_offdiag": float(action_abs_offdiag),
    }


def resolve_finite_match_kappas(
    cfg: EFTFiniteOneLoopMatchConfig,
    g_uv: np.ndarray | None = None,
    p_kin: np.ndarray | None = None,
    m2: np.ndarray | None = None,
    c_tree: np.ndarray | None = None,
) -> dict[str, float]:
    mode = str(cfg.mode).strip().lower()
    if mode not in {"constant", "input_tied", "action_normalized", "action_absolute"}:
        raise ValueError(f"Unsupported finite-match mode '{cfg.mode}'.")

    shell_spread = 0.0
    coeff_cv = 0.0
    offdiag_mix = 0.0
    gap_cv = 0.0
    gap_asym = 0.0
    g_col_norm_cv = 0.0
    c_tree_diag_cv = 0.0
    pkin_entropy = 0.0
    action_norm_diag = 1.0
    action_norm_offdiag = 1.0
    coeff_l1 = 0.0
    coeff_l2 = 0.0
    coeff_align = 0.0
    action_abs_diag = 0.0
    action_abs_offdiag = 0.0
    if mode in {"input_tied", "action_normalized", "action_absolute"}:
        if g_uv is None or p_kin is None or m2 is None or c_tree is None:
            raise ValueError(f"{mode} finite matching requires g_uv, p_kin, m2, and c_tree.")
        inv = finite_match_invariants(g_uv=g_uv, p_kin=p_kin, m2=m2, c_tree=c_tree, floor=cfg.floor)
        shell_spread = float(inv["shell_spread"])
        coeff_cv = float(inv["coeff_cv"])
        offdiag_mix = float(inv["offdiag_mix"])
    if mode in {"action_normalized", "action_absolute"}:
        a_inv = parent_action_invariants(g_uv=g_uv, p_kin=p_kin, m2=m2, c_tree=c_tree, floor=cfg.floor)
        gap_cv = float(a_inv["gap_cv"])
        gap_asym = float(a_inv["gap_asym"])
        g_col_norm_cv = float(a_inv["g_col_norm_cv"])
        c_tree_diag_cv = float(a_inv["c_tree_diag_cv"])
        pkin_entropy = float(a_inv["pkin_entropy"])
        action_norm_diag = float(a_inv["action_norm_diag"])
        action_norm_offdiag = float(a_inv["action_norm_offdiag"])
        coeff_l1 = float(a_inv["coeff_l1"])
        coeff_l2 = float(a_inv["coeff_l2"])
        coeff_align = float(a_inv["coeff_align"])
        action_abs_diag = float(a_inv["action_abs_diag"])
        action_abs_offdiag = float(a_inv["action_abs_offdiag"])

    kappa_diag_eff = float(cfg.kappa_diag)
    kappa_offdiag_eff = float(cfg.kappa_offdiag)
    if mode == "input_tied":
        kappa_diag_eff += float(cfg.input_diag_scale) * shell_spread * (1.0 + coeff_cv)
        kappa_offdiag_eff += float(cfg.input_offdiag_scale) * shell_spread * offdiag_mix
    elif mode == "action_normalized":
        kappa_diag_eff += (
            float(cfg.input_diag_scale)
            * shell_spread
            * (1.0 + coeff_cv)
            * action_norm_diag
        )
        kappa_offdiag_eff += (
            float(cfg.input_offdiag_scale)
            * shell_spread
            * offdiag_mix
            * action_norm_offdiag
        )
    elif mode == "action_absolute":
        kappa_diag_eff += (
            action_abs_diag
            * shell_spread
            * (1.0 + coeff_cv)
            * action_norm_diag
        )
        kappa_offdiag_eff += (
            action_abs_offdiag
            * shell_spread
            * offdiag_mix
            * action_norm_offdiag
        )

    return {
        "mode": mode,
        "kappa_diag_eff": float(kappa_diag_eff),
        "kappa_offdiag_eff": float(kappa_offdiag_eff),
        "shell_spread": float(shell_spread),
        "coeff_cv": float(coeff_cv),
        "offdiag_mix": float(offdiag_mix),
        "gap_cv": float(gap_cv),
        "gap_asym": float(gap_asym),
        "g_col_norm_cv": float(g_col_norm_cv),
        "c_tree_diag_cv": float(c_tree_diag_cv),
        "pkin_entropy": float(pkin_entropy),
        "action_norm_diag": float(action_norm_diag),
        "action_norm_offdiag": float(action_norm_offdiag),
        "coeff_l1": float(coeff_l1),
        "coeff_l2": float(coeff_l2),
        "coeff_align": float(coeff_align),
        "action_abs_diag": float(action_abs_diag),
        "action_abs_offdiag": float(action_abs_offdiag),
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
        "gap_cv": float(resolved["gap_cv"]),
        "gap_asym": float(resolved["gap_asym"]),
        "g_col_norm_cv": float(resolved["g_col_norm_cv"]),
        "c_tree_diag_cv": float(resolved["c_tree_diag_cv"]),
        "pkin_entropy": float(resolved["pkin_entropy"]),
        "action_norm_diag": float(resolved["action_norm_diag"]),
        "action_norm_offdiag": float(resolved["action_norm_offdiag"]),
        "coeff_l1": float(resolved["coeff_l1"]),
        "coeff_l2": float(resolved["coeff_l2"]),
        "coeff_align": float(resolved["coeff_align"]),
        "action_abs_diag": float(resolved["action_abs_diag"]),
        "action_abs_offdiag": float(resolved["action_abs_offdiag"]),
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
        gap_cv=float(meta["gap_cv"]),
        gap_asym=float(meta["gap_asym"]),
        g_col_norm_cv=float(meta["g_col_norm_cv"]),
        c_tree_diag_cv=float(meta["c_tree_diag_cv"]),
        pkin_entropy=float(meta["pkin_entropy"]),
        action_norm_diag=float(meta["action_norm_diag"]),
        action_norm_offdiag=float(meta["action_norm_offdiag"]),
        coeff_l1=float(meta["coeff_l1"]),
        coeff_l2=float(meta["coeff_l2"]),
        coeff_align=float(meta["coeff_align"]),
        action_abs_diag=float(meta["action_abs_diag"]),
        action_abs_offdiag=float(meta["action_abs_offdiag"]),
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
