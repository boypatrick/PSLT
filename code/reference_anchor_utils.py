#!/usr/bin/env python3
"""
Utilities for selecting and auditing the H->ll reference anchor (D0, eta0).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from hll_observable import HLLObservableConfig, HLLChannelPredictor
from pslt_lib import PSLTKinetics

try:
    from scipy.ndimage import distance_transform_edt  # type: ignore
except Exception:  # pragma: no cover - fallback path
    distance_transform_edt = None


def compute_r3_and_winner_maps(
    kinetics: PSLTKinetics,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    t_coh: float,
    n_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r3_map = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    winner_map = np.zeros_like(r3_map)
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            _, _, meta = kinetics.get_probabilities(float(d), float(eta), float(t_coh), N_max=int(n_max))
            r3_map[i, j] = float(meta["generation_ratio"])
            winner_map[i, j] = float(meta["winner"])
    return r3_map, winner_map


def compute_mu_chi2_maps(
    kinetics: PSLTKinetics,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    mode: str,
    t_coh: float,
    n_max: int,
    ref_d: float,
    ref_eta: float,
    mu_obs: float,
    sigma_obs: float,
    layer_n: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = HLLObservableConfig(
        mode=str(mode),
        t_coh=float(t_coh),
        ref_D=float(ref_d),
        ref_eta=float(ref_eta),
        n_max=int(n_max),
    )
    predictor = HLLChannelPredictor(kinetics, layer_n=layer_n, cfg=cfg)

    mu_map = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            mu_map[i, j] = float(predictor.mu_pred(float(d), float(eta)))
    chi2_map = ((mu_map - float(mu_obs)) / float(sigma_obs)) ** 2
    return mu_map, chi2_map


def _nearest_grid_index(d_vals: np.ndarray, eta_vals: np.ndarray, d0: float, eta0: float) -> Tuple[int, int]:
    i = int(np.argmin(np.abs(eta_vals - float(eta0))))
    j = int(np.argmin(np.abs(d_vals - float(d0))))
    return i, j


def _distance_to_boundary(mask: np.ndarray, d_step: float, eta_step: float) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=float)

    if distance_transform_edt is not None:
        return np.asarray(distance_transform_edt(mask.astype(np.uint8), sampling=(eta_step, d_step)), dtype=float)

    # Fallback: explicit nearest-outside-point distance (small 60x60 maps are fine).
    dist = np.zeros_like(mask, dtype=float)
    outside = np.argwhere(~mask)
    if outside.size == 0:
        dist[mask] = float("inf")
        return dist

    outside_xy = np.column_stack((outside[:, 0] * eta_step, outside[:, 1] * d_step))
    inside = np.argwhere(mask)
    for i, j in inside:
        p = np.array([i * eta_step, j * d_step], dtype=float)
        diff = outside_xy - p
        dist[i, j] = float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))
    return dist


def _candidate_row(
    mode: str,
    i: int,
    j: int,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    chi2_map: np.ndarray,
    r3_map: np.ndarray,
    boundary_dist: np.ndarray,
) -> Dict[str, float | str]:
    chi2_val = float(chi2_map[i, j])
    return {
        "mode": mode,
        "ref_D": float(d_vals[j]),
        "ref_eta": float(eta_vals[i]),
        "grid_i": int(i),
        "grid_j": int(j),
        "chi2_at_point": chi2_val,
        "r3_at_point": float(r3_map[i, j]),
        "in_acceptance": int(chi2_val <= 4.0),
        "boundary_distance": float(boundary_dist[i, j]),
    }


def select_anchor_candidates_from_fixed_scan(
    kinetics: PSLTKinetics,
    d_vals: np.ndarray,
    eta_vals: np.ndarray,
    mode: str,
    t_coh: float,
    n_max: int,
    mu_obs: float,
    sigma_obs: float,
    fixed_ref_d: float,
    fixed_ref_eta: float,
) -> Dict[str, Dict[str, float | str]]:
    r3_map, _ = compute_r3_and_winner_maps(kinetics, d_vals, eta_vals, t_coh=float(t_coh), n_max=int(n_max))
    _, chi2_map = compute_mu_chi2_maps(
        kinetics,
        d_vals,
        eta_vals,
        mode=str(mode),
        t_coh=float(t_coh),
        n_max=int(n_max),
        ref_d=float(fixed_ref_d),
        ref_eta=float(fixed_ref_eta),
        mu_obs=float(mu_obs),
        sigma_obs=float(sigma_obs),
        layer_n=2,
    )

    if len(d_vals) > 1:
        d_step = float(np.mean(np.diff(d_vals)))
    else:
        d_step = 1.0
    if len(eta_vals) > 1:
        eta_step = float(np.mean(np.diff(eta_vals)))
    else:
        eta_step = 1.0

    accept_mask = chi2_map <= 4.0
    boundary_dist = _distance_to_boundary(accept_mask, d_step=d_step, eta_step=eta_step)

    fixed_i, fixed_j = _nearest_grid_index(d_vals, eta_vals, fixed_ref_d, fixed_ref_eta)

    best_i, best_j = np.unravel_index(int(np.argmin(chi2_map)), chi2_map.shape)

    if np.any(accept_mask):
        max_dist = float(np.max(boundary_dist[accept_mask]))
        robust_mask = accept_mask & np.isclose(boundary_dist, max_dist, rtol=0.0, atol=1e-12)
        robust_idx = np.argwhere(robust_mask)
        # Tie-break: highest R3, then lowest chi2.
        robust_rank = sorted(
            robust_idx.tolist(),
            key=lambda ij: (-float(r3_map[ij[0], ij[1]]), float(chi2_map[ij[0], ij[1]])),
        )
        robust_i, robust_j = int(robust_rank[0][0]), int(robust_rank[0][1])
    else:
        robust_i, robust_j = int(best_i), int(best_j)

    candidates: Dict[str, Dict[str, float | str]] = {
        "fixed": _candidate_row("fixed", fixed_i, fixed_j, d_vals, eta_vals, chi2_map, r3_map, boundary_dist),
        "chi2_best": _candidate_row("chi2_best", int(best_i), int(best_j), d_vals, eta_vals, chi2_map, r3_map, boundary_dist),
        "robust_center": _candidate_row(
            "robust_center", int(robust_i), int(robust_j), d_vals, eta_vals, chi2_map, r3_map, boundary_dist
        ),
        "meta": {
            "fixed_scan_ref_D": float(fixed_ref_d),
            "fixed_scan_ref_eta": float(fixed_ref_eta),
            "f_chi2_le_4": float(np.mean(accept_mask)),
            "best_chi2": float(np.min(chi2_map)),
            "best_D": float(d_vals[int(best_j)]),
            "best_eta": float(eta_vals[int(best_i)]),
            "r3_mean": float(np.mean(r3_map)),
            "r3_frac_gt_090": float(np.mean(r3_map >= 0.90)),
        },
    }
    return candidates

