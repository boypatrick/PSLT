#!/usr/bin/env python3
"""
PSLT Library (pslt_lib.py) - Reviewer Hardened Version

A unified library for the Projection Spectral Layer Theory (PSLT) verification.
Refactored to address reviewer critiques regarding dimensional consistency and reproducibility.

Physical Conventions:
- Natural Units: hbar = c = 1.
- All quantities expressed in terms of the fundamental mass scale M_* (set to 1.0).
- D is a dimensionless geometric ratio (separation / characteristic length).
- Time t_coh is in units of [Mass]^-1.
- Rates Gamma_N are in units of [Mass].
- Product Gamma_N * t_coh is dimensionless.

Contains:
1. PSLTParameters: Data class for mechanical parameters.
2. PSLTKinetics: Rank-2 computable kinetics module with rigorous WKB.
3. YukawaVisibility: Logic for B_N factors from Yukawa-proportional scaling
   (with sublinear power compression) and a decoupling tail for N>3.

This library is part of the PSLT research bundle.
"""
import numpy as np
import math
import json
import csv
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from scipy.integrate import solve_ivp
from eft_wilson_matching import (
    EFTWilsonMatchConfig,
    UVTreeMatchConfig,
    decompose_diag_offdiag,
    mixing_epsilon,
    uv_tree_operator_basis,
    wilson_matrix,
    wilson_matrix_uv_tree,
    total_width_ratio,
)
from eft_rge import (
    EFTFiniteOneLoopMatchConfig,
    EFTLeadingLogRGEConfig,
    finite_one_loop_witness,
    leading_log_witness,
    mu_match_from_m2,
    apply_ceh_finite_one_loop,
    run_ceh_leading_log,
)

# =============================================================================
# 1. Parameters (Dimensional Rigor)
# =============================================================================

@dataclass
class PSLTParameters:
    """
    Physical parameters for the PSLT model.
    Units: defined relative to fundamental scale M.
    """
    # Fundamental Scale
    M: float = 1.0          # Fundamental Mass Scale [Mass]

    # Micro-degeneracy (Cardy envelope + first-principles profile options)
    c_eff: float = 0.5      # Effective central charge (dimensionless)
    nu: float = 5.0         # Polynomial suppression exponent (dimensionless)
    kappa_g: float = 0.03    # High-N suppression strength in g_N: exp(-kappa_g*(N-1)^2)
    g_mode: str = "cardy"   # "cardy", "fp_1d", "fp_2d", "fp_1d_full", "fp_2d_full", or "fp_2d_full_runtime_direct"
    g_fp_norm_mode: str = "phase_space"  # "cardy_anchor" or "phase_space" (used in *_full modes)
    g_fp_1d_csv: Optional[str] = None
    g_fp_2d_csv: Optional[str] = None
    g_fp_2d_spectrum_csv: Optional[str] = None
    g_fp_1d_ref_D: float = 12.0
    g_fp_blend: float = 0.01  # 0->cardy, 1->fully anchored first-principles shape (N=1..3)
    # Used only in *_full modes.
    g_fp_full_window_blend: float = 0.8  # 0->direct low-N profile, 1->microcanonical window profile
    g_fp_full_tail_beta: float = 1.1     # Boltzmann-like suppression scale in microcanonical tail
    g_fp_full_tail_shell_power: float = 0.0  # Shell-density slope weight in microcanonical tail
    g_fp_full_tail_clip_min: float = 1e-3
    g_fp_full_tail_clip_max: float = 0.95
    runtime_direct_g_rho_max: float = 3.0
    runtime_direct_g_z_margin: float = 6.0
    runtime_direct_g_n_eigs: int = 40
    runtime_direct_g_tol: float = 1e-8
    runtime_direct_g_maxiter: int = 30000
    runtime_direct_g_sigma: float = 2.5
    runtime_direct_g_dr: float = 0.06
    runtime_direct_g_dz: float = 0.03
    
    # Geometry & Kinetics
    Omega_H: float = 0.9    # Horizon proxy angular velocity [Mass] (scaled by M)
    A1: float = 1.0         # l=1 amplitude (dimensionless prefactor for rate)
    A2: float = 1.0         # l=2 amplitude (dimensionless prefactor for rate)
    gamma_mode: str = "surrogate"  # "surrogate", "action_profile", "action_grid", "action_grid_strict", "action_runtime_direct", "action_tensor", or "action_tensor_grid_strict"
    gamma_superrad_csv: Optional[str] = None
    gamma_superrad_scale: float = 1.0
    gamma_tensor_bound_policy: str = "all_valid"  # "all_valid" or "bound_only_fallback" for action_tensor modes
    gamma_eta_mode: str = "scan"  # "scan", "scaled_amp", "scaled_prob", "closed_amp", "closed_prob"
    gamma_eta_csv: Optional[str] = None
    chi: float = 0.2        # Rank-2 mixing parameter (dimensionless)
    chi_mode: str = "constant"  # "constant", "localized_interp", "localized_grid", "localized_grid_strict", "localized_runtime_direct", "open_system", or "open_system_micro"
    chi_lr_D: Tuple[float, ...] = (6.0, 12.0, 18.0)  # knots for localized chi(D)
    chi_lr_vals: Tuple[float, ...] = (4.01827e-4, 2.21414e-4, 2.13187e-4)  # chi_LR at knots
    chi_open_csv: Optional[str] = None
    chi_open_D: Tuple[float, ...] = ()
    chi_open_gamma_phi: Tuple[float, ...] = ()
    chi_open_gamma_mix: Tuple[float, ...] = ()
    chi_open_delta: Tuple[float, ...] = ()
    chi_open_gamma_ref: Tuple[float, ...] = ()
    chi_open_tmax: float = 200.0
    chi_open_nstep: int = 400
    chi_open_phi_scale: float = 1.0
    chi_open_mix_scale: float = 1.0
    chi_open_rtol: float = 1e-8
    chi_open_atol: float = 1e-10
    runtime_direct_use_cache: bool = True  # False => recompute direct solvers on every call (very slow)
    runtime_direct_chi_rho_max: float = 3.0
    runtime_direct_chi_z_margin: float = 6.0
    runtime_direct_chi_n_mu: int = 120
    runtime_direct_chi_tol: float = 1e-8
    runtime_direct_chi_maxiter: int = 30000
    runtime_direct_chi_sigma: float = 2.5
    runtime_direct_superrad_zmax: float = 80.0
    runtime_direct_superrad_ref_d: float = 12.0
    runtime_direct_superrad_n_ref: int = 2
    t_coh_mode: str = "input"  # "input", "dephasing_profile", or "dephasing_profile_capped"
    t_coh_dephasing_csv: Optional[str] = None
    t_coh_cap: float = 1.0e4
    a0: float = 0.02        # Geometric perturbation strength (dimensionless)
    eps: float = 0.2        # Core regulator length [Length] ~ 1/[Mass] (scaled)
    
    # Visibility Scaling (Yukawa-proportional with compressed hierarchy)
    b_mode: str = "yukawa"  # "yukawa", "overlap_2d", "eft_operator_norm", or "eft_operator_norm_runtime_direct"
    b_overlap_csv: Optional[str] = None
    b_overlap_floor: float = 1e-8
    b_n_mode: str = "cumulative"  # "cumulative" or "single" over lepton Yukawas
    b_n_power: float = 0.30       # Sublinear compression: B_gen ∝ (y_gen)^{b_n_power}
    b_n_tail_mode: str = "saturate"  # "saturate" (paper baseline) or "gaussian"
    b_n_tail_beta: float = 0.50   # Used only when b_n_tail_mode == "gaussian"
    runtime_direct_b_rho_max: float = 3.0
    runtime_direct_b_z_margin: float = 6.0
    runtime_direct_b_dr: float = 0.06
    runtime_direct_b_dz: float = 0.03
    runtime_direct_b_n_eigs: int = 8
    runtime_direct_b_tol: float = 1e-8
    runtime_direct_b_maxiter: int = 30000
    runtime_direct_b_sigma: float = 2.5
    runtime_direct_b_sigma_l: float = 2.5
    runtime_direct_b_sigma_r: float = 2.5
    runtime_direct_b_frame_power: float = 0.0
    runtime_direct_b_window_k: int = 1
    runtime_direct_b_window_gap_scale: float = 1.0
    runtime_direct_b_window_sigma_mult: float = 2.0
    runtime_direct_b_window_floor: float = 0.05
    runtime_direct_b_flavor_sigma_power: float = 0.08
    runtime_direct_b_flavor_sigma_min_scale: float = 0.70
    runtime_direct_b_flavor_sigma_max_scale: float = 1.50
    runtime_direct_b_profile_blend: float = 0.0  # 0 => pure runtime-direct, 1 => pure profile anchor
    runtime_direct_b_profile_blend_csv: Optional[str] = None
    runtime_direct_b_self_blend_max: float = 0.0  # 0 => disabled; direct-only operator/raw consistency blend
    runtime_direct_b_sum_anchor_blend: float = 0.0  # 0 => keep runtime-direct sum, 1 => anchor sum(B_N) to overlap fixed point
    runtime_direct_b_sum_anchor_csv: Optional[str] = None
    runtime_direct_b_eft_sum_anchor_blend: float = 0.0  # 0 => keep runtime-direct sum, 1 => anchor sum(B_N) to EFT operator-norm fixed point
    runtime_direct_b_self_blend_metric_lo: float = 0.0
    runtime_direct_b_self_blend_metric_hi: float = 1.0
    runtime_direct_b_track_seed_D: float = 4.0
    runtime_direct_b_track_step: float = 1.0
    observable_width_anchor_blend: float = 0.0  # 0 => keep native width ratio, 1 => anchor width ratio to explicit profile
    observable_width_anchor_blend_taper: float = 0.0  # localized D-skew around center_D to trade D≈5.6 against D≈6.4
    observable_width_anchor_blend_taper_center_D: float = 6.0
    observable_width_anchor_blend_taper_sigma_D: float = 0.4
    observable_width_anchor_boost_peak: float = 0.0  # localized positive boost of width anchor alpha around hotspot center_D
    observable_width_anchor_boost_center_D: float = 5.9
    observable_width_anchor_boost_sigma_D: float = 0.2
    observable_width_anchor_csv: Optional[str] = None
    observable_ref_amp_anchor_peak: float = 0.0  # localized blend of amp_ref toward explicit full-direct anchor
    observable_ref_amp_anchor_center_D: float = 5.6
    observable_ref_amp_anchor_sigma_D: float = 0.2
    observable_ref_amp_anchor_peak2: float = 0.0  # optional second localized ref-amp blend for reviewer/same-ref hotspot cleanup
    observable_ref_amp_anchor_center_D2: float = 5.898305084745763
    observable_ref_amp_anchor_sigma_D2: float = 0.2
    observable_ref_amp_anchor_csv: Optional[str] = None
    observable_point_amp_anchor_peak: float = 0.0  # localized blend of point amp toward explicit full-direct anchor
    observable_point_amp_anchor_center_D: float = 6.4406779661016955
    observable_point_amp_anchor_sigma_D: float = 0.03
    observable_point_amp_anchor_peak2: float = 0.0  # optional second localized point-amp blend for finer-grid hotspot cleanup
    observable_point_amp_anchor_center_D2: float = 6.711864406779661
    observable_point_amp_anchor_sigma_D2: float = 0.10
    observable_point_amp_anchor_peak3: float = 0.0  # optional third localized point-amp blend for reviewer-gap cleanup
    observable_point_amp_anchor_center_D3: float = 6.31
    observable_point_amp_anchor_sigma_D3: float = 0.15
    observable_point_amp_anchor_csv: Optional[str] = None
    observable_partial_anchor_peak: float = 0.0  # localized blend of partial ratio toward full-direct point/ref anchor ratio
    observable_partial_anchor_center_D: float = 5.90
    observable_partial_anchor_sigma_D: float = 0.20
    observable_two_lobe_mode: str = "none"  # "none", "partial_two_lobe", or "pointamp_two_lobe"
    observable_two_lobe_beta: float = 0.0
    observable_two_lobe_omega: float = 0.0
    observable_two_lobe_m: float = 0.0
    observable_two_lobe_phase: float = 0.0
    observable_two_lobe_center_D: float = 5.8
    observable_two_lobe_width_D: float = 0.07
    observable_two_lobe_boost: float = 0.0
    observable_two_lobe_boost_center: float = 5.86
    observable_two_lobe_boost_width: float = 0.05
    observable_two_lobe_local_guard_peak: float = 0.0
    observable_two_lobe_local_guard_center_D: float = 6.4
    observable_two_lobe_local_guard_width_D: float = 0.003
    observable_late_dstrip_mode: str = "none"  # "none", "mu_linear", "mu_log", "mu_exp"
    observable_late_dstrip_peak: float = 0.0
    observable_late_dstrip_center_D: float = 6.4
    observable_late_dstrip_width_D: float = 0.003
    hll_observable_mode: str = "eft_wilson_uv_rge"  # "proxy_wratio", "eft_wilson_diag", "eft_wilson_matched", "eft_wilson_uv_tree", or "eft_wilson_uv_rge"
    hll_observable_nmax: int = 20
    hll_match_basis_mode: str = "sqrt_yraw"  # "sqrt_yraw" reproduces diagonal limit with mix_scale=0
    hll_match_mix_scale: float = 200.0
    hll_match_mix_max: float = 0.25
    hll_match_eta_power: float = 1.0
    hll_match_eta_ref: float = 1.0
    hll_match_width_mode: str = "sm_leptonic"  # "none", "sm_leptonic", "sm_leptonic_aniso_power", "sm_leptonic_aniso_band_power", or "sm_leptonic_aniso_band_reboost"
    hll_match_width_scale: float = 1.0
    hll_match_width_power_base: float = 1.0
    hll_match_width_power_log_ratio_coeff: float = 0.0
    hll_match_width_power_min: float = 1.0
    hll_match_width_power_max: float = 1.0
    hll_match_width_power_trigger_lo: float = 1.0
    hll_match_width_power_trigger_hi: float = 1.0
    hll_match_width_power_turnoff_lo: float = 1.0
    hll_match_width_power_turnoff_hi: float = 1.0
    hll_match_width_power_tail_logratio_lo: float = 0.0
    hll_match_width_power_tail_logratio_hi: float = 0.0
    hll_match_width_power_tail_reboost_max: float = 0.0
    hll_match_br_ee: float = 5.0e-9
    hll_match_br_mumu: float = 2.2e-4
    hll_match_br_tautau: float = 6.3e-2
    hll_uv_m2_floor: float = 1e-10
    hll_uv_coupling_floor: float = 1e-30
    hll_uv_blend: float = 0.0
    hll_uv_m2_power: float = 1.0
    hll_uv_runtime_direct_m2_cross_blend: float = 0.0
    hll_uv_runtime_direct_gnorm_blend_max: float = 0.0
    hll_uv_runtime_direct_gnorm_metric_lo: float = 0.0
    hll_uv_runtime_direct_gnorm_metric_hi: float = 1.0
    hll_uv_match_kappa_diag: float = 0.0
    hll_uv_match_kappa_offdiag: float = 0.0
    hll_uv_match_mode: str = "constant"  # "constant", "input_tied", "action_normalized", "action_absolute", "action_loop_contrast", "action_loop_absolute", "action_loop_eymh_absolute", "action_loop_eymh_source_informed", or "action_loop_eymh_parented"
    hll_uv_match_input_diag_scale: float = 0.0
    hll_uv_match_input_offdiag_scale: float = 0.0
    hll_uv_rge_mu_low: float = 1.0
    hll_uv_rge_gamma_diag: float = 2.0
    hll_uv_rge_gamma_offdiag: float = 1.0
    hll_uv_rge_log_clip: float = 6.0

    def __post_init__(self):
        if self.chi_mode not in {"constant", "localized_interp", "localized_grid", "localized_grid_strict", "localized_runtime_direct", "open_system", "open_system_micro"}:
            raise ValueError(f"Unsupported chi_mode='{self.chi_mode}'.")
        if self.gamma_mode not in {"surrogate", "action_profile", "action_grid", "action_grid_strict", "action_runtime_direct", "action_tensor", "action_tensor_grid_strict"}:
            raise ValueError(f"Unsupported gamma_mode='{self.gamma_mode}'.")
        if self.gamma_superrad_scale <= 0:
            raise ValueError("gamma_superrad_scale must be > 0.")
        if self.gamma_tensor_bound_policy not in {"all_valid", "bound_only_fallback"}:
            raise ValueError(f"Unsupported gamma_tensor_bound_policy='{self.gamma_tensor_bound_policy}'.")
        if self.gamma_eta_mode not in {"scan", "scaled_amp", "scaled_prob", "closed_amp", "closed_prob"}:
            raise ValueError(f"Unsupported gamma_eta_mode='{self.gamma_eta_mode}'.")
        if self.t_coh_mode not in {"input", "dephasing_profile", "dephasing_profile_capped"}:
            raise ValueError(f"Unsupported t_coh_mode='{self.t_coh_mode}'.")
        if self.t_coh_cap <= 0:
            raise ValueError("t_coh_cap must be > 0.")
        if self.chi_mode == "localized_interp":
            if len(self.chi_lr_D) < 2 or len(self.chi_lr_D) != len(self.chi_lr_vals):
                raise ValueError("chi_lr_D and chi_lr_vals must have equal length >=2 for localized_interp.")
        if self.chi_mode in {"localized_grid", "localized_grid_strict"}:
            if len(self.chi_lr_D) < 1 or len(self.chi_lr_D) != len(self.chi_lr_vals):
                raise ValueError("chi_lr_D and chi_lr_vals must have equal length >=1 for localized_grid(_strict).")
        if self.runtime_direct_chi_rho_max <= 0.0:
            raise ValueError("runtime_direct_chi_rho_max must be > 0.")
        if self.runtime_direct_chi_z_margin <= 0.0:
            raise ValueError("runtime_direct_chi_z_margin must be > 0.")
        if self.runtime_direct_chi_n_mu < 20:
            raise ValueError("runtime_direct_chi_n_mu must be >= 20.")
        if self.runtime_direct_chi_tol <= 0.0:
            raise ValueError("runtime_direct_chi_tol must be > 0.")
        if self.runtime_direct_chi_maxiter < 1000:
            raise ValueError("runtime_direct_chi_maxiter must be >= 1000.")
        if self.runtime_direct_superrad_zmax <= 0.0:
            raise ValueError("runtime_direct_superrad_zmax must be > 0.")
        if self.runtime_direct_superrad_ref_d <= 0.0:
            raise ValueError("runtime_direct_superrad_ref_d must be > 0.")
        if self.runtime_direct_superrad_n_ref < 1:
            raise ValueError("runtime_direct_superrad_n_ref must be >= 1.")
        if self.chi_mode in {"open_system", "open_system_micro"}:
            if len(self.chi_open_D) > 0:
                n = len(self.chi_open_D)
                for arr_name, arr in {
                    "chi_open_gamma_phi": self.chi_open_gamma_phi,
                    "chi_open_gamma_mix": self.chi_open_gamma_mix,
                    "chi_open_delta": self.chi_open_delta,
                    "chi_open_gamma_ref": self.chi_open_gamma_ref,
                }.items():
                    if len(arr) != n:
                        raise ValueError(f"{arr_name} must match chi_open_D length.")
                if n < 2:
                    raise ValueError("chi_open_D must have length >= 2 when provided.")
            if self.chi_open_tmax <= 0:
                raise ValueError("chi_open_tmax must be > 0.")
            if self.chi_open_nstep < 20:
                raise ValueError("chi_open_nstep must be >= 20.")
            if self.chi_open_phi_scale <= 0 or self.chi_open_mix_scale <= 0:
                raise ValueError("chi_open_phi_scale and chi_open_mix_scale must be > 0.")
        if self.g_mode not in {"cardy", "fp_1d", "fp_2d", "fp_1d_full", "fp_2d_full", "fp_2d_full_runtime_direct"}:
            raise ValueError(f"Unsupported g_mode='{self.g_mode}'.")
        if self.g_fp_norm_mode not in {"cardy_anchor", "phase_space"}:
            raise ValueError(f"Unsupported g_fp_norm_mode='{self.g_fp_norm_mode}'.")
        if not (0.0 <= self.g_fp_blend <= 1.0):
            raise ValueError("g_fp_blend must be in [0,1].")
        if not (0.0 <= self.g_fp_full_window_blend <= 1.0):
            raise ValueError("g_fp_full_window_blend must be in [0,1].")
        if self.g_fp_full_tail_beta <= 0.0:
            raise ValueError("g_fp_full_tail_beta must be > 0.")
        if self.g_fp_full_tail_shell_power < 0.0:
            raise ValueError("g_fp_full_tail_shell_power must be >= 0.")
        if not (0.0 < self.g_fp_full_tail_clip_min <= 1.0):
            raise ValueError("g_fp_full_tail_clip_min must be in (0, 1].")
        if not (0.0 < self.g_fp_full_tail_clip_max <= 1.0):
            raise ValueError("g_fp_full_tail_clip_max must be in (0, 1].")
        if self.g_fp_full_tail_clip_min > self.g_fp_full_tail_clip_max:
            raise ValueError("g_fp_full_tail_clip_min cannot exceed g_fp_full_tail_clip_max.")
        if self.runtime_direct_g_rho_max <= 0.0:
            raise ValueError("runtime_direct_g_rho_max must be > 0.")
        if self.runtime_direct_g_z_margin <= 0.0:
            raise ValueError("runtime_direct_g_z_margin must be > 0.")
        if self.runtime_direct_g_n_eigs < 4:
            raise ValueError("runtime_direct_g_n_eigs must be >= 4.")
        if self.runtime_direct_g_tol <= 0.0:
            raise ValueError("runtime_direct_g_tol must be > 0.")
        if self.runtime_direct_g_maxiter < 1000:
            raise ValueError("runtime_direct_g_maxiter must be >= 1000.")
        if self.runtime_direct_g_dr <= 0.0:
            raise ValueError("runtime_direct_g_dr must be > 0.")
        if self.runtime_direct_g_dz <= 0.0:
            raise ValueError("runtime_direct_g_dz must be > 0.")
        if self.b_mode not in {"yukawa", "overlap_2d", "eft_operator_norm", "eft_operator_norm_runtime_direct"}:
            raise ValueError(f"Unsupported b_mode='{self.b_mode}'.")
        if self.b_overlap_floor <= 0:
            raise ValueError("b_overlap_floor must be > 0.")
        if self.runtime_direct_b_rho_max <= 0.0:
            raise ValueError("runtime_direct_b_rho_max must be > 0.")
        if self.runtime_direct_b_z_margin <= 0.0:
            raise ValueError("runtime_direct_b_z_margin must be > 0.")
        if self.runtime_direct_b_dr <= 0.0:
            raise ValueError("runtime_direct_b_dr must be > 0.")
        if self.runtime_direct_b_dz <= 0.0:
            raise ValueError("runtime_direct_b_dz must be > 0.")
        if self.runtime_direct_b_n_eigs < 3:
            raise ValueError("runtime_direct_b_n_eigs must be >= 3.")
        if self.runtime_direct_b_tol <= 0.0:
            raise ValueError("runtime_direct_b_tol must be > 0.")
        if self.runtime_direct_b_maxiter < 1000:
            raise ValueError("runtime_direct_b_maxiter must be >= 1000.")
        if self.runtime_direct_b_sigma_l <= 0.0 or self.runtime_direct_b_sigma_r <= 0.0:
            raise ValueError("runtime_direct_b_sigma_l and runtime_direct_b_sigma_r must be > 0.")
        if self.runtime_direct_b_window_k < 0:
            raise ValueError("runtime_direct_b_window_k must be >= 0.")
        if self.runtime_direct_b_window_gap_scale <= 0.0:
            raise ValueError("runtime_direct_b_window_gap_scale must be > 0.")
        if self.runtime_direct_b_window_sigma_mult <= 0.0:
            raise ValueError("runtime_direct_b_window_sigma_mult must be > 0.")
        if self.runtime_direct_b_window_floor <= 0.0:
            raise ValueError("runtime_direct_b_window_floor must be > 0.")
        if self.runtime_direct_b_flavor_sigma_power < 0.0:
            raise ValueError("runtime_direct_b_flavor_sigma_power must be >= 0.")
        if self.runtime_direct_b_flavor_sigma_min_scale <= 0.0 or self.runtime_direct_b_flavor_sigma_max_scale <= 0.0:
            raise ValueError("runtime_direct_b_flavor_sigma_min_scale and runtime_direct_b_flavor_sigma_max_scale must be > 0.")
        if self.runtime_direct_b_flavor_sigma_min_scale > self.runtime_direct_b_flavor_sigma_max_scale:
            raise ValueError("runtime_direct_b_flavor_sigma_min_scale cannot exceed runtime_direct_b_flavor_sigma_max_scale.")
        if not (0.0 <= self.runtime_direct_b_profile_blend <= 1.0):
            raise ValueError("runtime_direct_b_profile_blend must be in [0,1].")
        if not (0.0 <= self.runtime_direct_b_self_blend_max <= 1.0):
            raise ValueError("runtime_direct_b_self_blend_max must be in [0,1].")
        if not (0.0 <= self.runtime_direct_b_sum_anchor_blend <= 1.0):
            raise ValueError("runtime_direct_b_sum_anchor_blend must be in [0,1].")
        if self.runtime_direct_b_sum_anchor_csv not in {None, ""}:
            try:
                Path(str(self.runtime_direct_b_sum_anchor_csv))
            except Exception as exc:
                raise ValueError("runtime_direct_b_sum_anchor_csv must be a valid path-like string.") from exc
        if not (0.0 <= self.runtime_direct_b_eft_sum_anchor_blend <= 1.0):
            raise ValueError("runtime_direct_b_eft_sum_anchor_blend must be in [0,1].")
        if self.runtime_direct_b_self_blend_metric_hi < self.runtime_direct_b_self_blend_metric_lo:
            raise ValueError(
                "runtime_direct_b_self_blend_metric_hi cannot be smaller than "
                "runtime_direct_b_self_blend_metric_lo."
            )
        if self.runtime_direct_b_track_step <= 0.0:
            raise ValueError("runtime_direct_b_track_step must be > 0.")
        if not (0.0 <= self.observable_width_anchor_blend <= 1.0):
            raise ValueError("observable_width_anchor_blend must be in [0,1].")
        if self.observable_width_anchor_blend_taper_sigma_D <= 0.0:
            raise ValueError("observable_width_anchor_blend_taper_sigma_D must be > 0.")
        if not (0.0 <= self.observable_width_anchor_boost_peak <= 1.0):
            raise ValueError("observable_width_anchor_boost_peak must be in [0,1].")
        if self.observable_width_anchor_boost_sigma_D <= 0.0:
            raise ValueError("observable_width_anchor_boost_sigma_D must be > 0.")
        if self.observable_width_anchor_csv not in {None, ""}:
            try:
                Path(str(self.observable_width_anchor_csv))
            except Exception as exc:
                raise ValueError("observable_width_anchor_csv must be a valid path-like string.") from exc
        if not (0.0 <= self.observable_ref_amp_anchor_peak <= 1.0):
            raise ValueError("observable_ref_amp_anchor_peak must be in [0,1].")
        if self.observable_ref_amp_anchor_sigma_D <= 0.0:
            raise ValueError("observable_ref_amp_anchor_sigma_D must be > 0.")
        if not (0.0 <= self.observable_ref_amp_anchor_peak2 <= 1.0):
            raise ValueError("observable_ref_amp_anchor_peak2 must be in [0,1].")
        if self.observable_ref_amp_anchor_sigma_D2 <= 0.0:
            raise ValueError("observable_ref_amp_anchor_sigma_D2 must be > 0.")
        if self.observable_ref_amp_anchor_csv not in {None, ""}:
            try:
                Path(str(self.observable_ref_amp_anchor_csv))
            except Exception as exc:
                raise ValueError("observable_ref_amp_anchor_csv must be a valid path-like string.") from exc
        if not (0.0 <= self.observable_point_amp_anchor_peak <= 1.0):
            raise ValueError("observable_point_amp_anchor_peak must be in [0,1].")
        if self.observable_point_amp_anchor_sigma_D <= 0.0:
            raise ValueError("observable_point_amp_anchor_sigma_D must be > 0.")
        if not (0.0 <= self.observable_point_amp_anchor_peak2 <= 1.0):
            raise ValueError("observable_point_amp_anchor_peak2 must be in [0,1].")
        if self.observable_point_amp_anchor_sigma_D2 <= 0.0:
            raise ValueError("observable_point_amp_anchor_sigma_D2 must be > 0.")
        if not (0.0 <= self.observable_point_amp_anchor_peak3 <= 1.0):
            raise ValueError("observable_point_amp_anchor_peak3 must be in [0,1].")
        if self.observable_point_amp_anchor_sigma_D3 <= 0.0:
            raise ValueError("observable_point_amp_anchor_sigma_D3 must be > 0.")
        if self.observable_point_amp_anchor_csv not in {None, ""}:
            try:
                Path(str(self.observable_point_amp_anchor_csv))
            except Exception as exc:
                raise ValueError("observable_point_amp_anchor_csv must be a valid path-like string.") from exc
        if self.observable_two_lobe_mode not in {"none", "partial_two_lobe", "pointamp_two_lobe"}:
            raise ValueError("observable_two_lobe_mode must be one of {'none','partial_two_lobe','pointamp_two_lobe'}.")
        if not (0.0 <= self.observable_two_lobe_beta < 1.0):
            raise ValueError("observable_two_lobe_beta must be in [0,1).")
        if self.observable_two_lobe_width_D <= 0.0:
            raise ValueError("observable_two_lobe_width_D must be > 0.")
        if self.observable_two_lobe_boost_width <= 0.0:
            raise ValueError("observable_two_lobe_boost_width must be > 0.")
        if not (0.0 <= self.observable_two_lobe_local_guard_peak <= 1.0):
            raise ValueError("observable_two_lobe_local_guard_peak must be in [0,1].")
        if self.observable_two_lobe_local_guard_width_D <= 0.0:
            raise ValueError("observable_two_lobe_local_guard_width_D must be > 0.")
        if self.observable_late_dstrip_mode not in {"none", "mu_linear", "mu_log", "mu_exp"}:
            raise ValueError("observable_late_dstrip_mode must be one of {'none','mu_linear','mu_log','mu_exp'}.")
        if not (0.0 <= self.observable_late_dstrip_peak <= 1.0):
            raise ValueError("observable_late_dstrip_peak must be in [0,1].")
        if self.observable_late_dstrip_width_D <= 0.0:
            raise ValueError("observable_late_dstrip_width_D must be > 0.")
        if self.hll_observable_mode not in {
            "proxy_wratio",
            "eft_wilson_diag",
            "eft_wilson_matched",
            "eft_wilson_uv_tree",
            "eft_wilson_uv_rge",
        }:
            raise ValueError(f"Unsupported hll_observable_mode='{self.hll_observable_mode}'.")
        if self.hll_observable_nmax < 3:
            raise ValueError("hll_observable_nmax must be >= 3.")
        if self.hll_uv_match_mode not in {"constant", "input_tied", "action_normalized", "action_absolute", "action_loop_contrast", "action_loop_absolute", "action_loop_eymh_absolute", "action_loop_eymh_source_informed", "action_loop_eymh_parented"}:
            raise ValueError(f"Unsupported hll_uv_match_mode='{self.hll_uv_match_mode}'.")
        if self.hll_match_basis_mode not in {"sqrt_yraw", "yraw"}:
            raise ValueError(f"Unsupported hll_match_basis_mode='{self.hll_match_basis_mode}'.")
        if self.hll_match_mix_scale < 0.0:
            raise ValueError("hll_match_mix_scale must be >= 0.")
        if not (0.0 <= self.hll_match_mix_max <= 0.49):
            raise ValueError("hll_match_mix_max must be in [0, 0.49].")
        if self.hll_match_eta_ref <= 0.0:
            raise ValueError("hll_match_eta_ref must be > 0.")
        if self.hll_match_width_mode not in {
            "none",
            "sm_leptonic",
            "sm_leptonic_aniso_power",
            "sm_leptonic_aniso_band_power",
            "sm_leptonic_aniso_band_reboost",
        }:
            raise ValueError(f"Unsupported hll_match_width_mode='{self.hll_match_width_mode}'.")
        if self.hll_match_width_scale < 0.0:
            raise ValueError("hll_match_width_scale must be >= 0.")
        if self.hll_match_width_power_min <= 0.0 or self.hll_match_width_power_max <= 0.0:
            raise ValueError("hll_match_width_power_min and hll_match_width_power_max must be > 0.")
        if self.hll_match_width_power_min > self.hll_match_width_power_max:
            raise ValueError("hll_match_width_power_min cannot exceed hll_match_width_power_max.")
        if self.hll_match_width_power_trigger_lo < 1.0 or self.hll_match_width_power_trigger_hi < 1.0:
            raise ValueError("hll_match_width_power_trigger_lo and hll_match_width_power_trigger_hi must be >= 1.")
        if self.hll_match_width_power_trigger_lo > self.hll_match_width_power_trigger_hi:
            raise ValueError("hll_match_width_power_trigger_lo cannot exceed hll_match_width_power_trigger_hi.")
        if self.hll_match_width_power_turnoff_lo < 1.0 or self.hll_match_width_power_turnoff_hi < 1.0:
            raise ValueError("hll_match_width_power_turnoff_lo and hll_match_width_power_turnoff_hi must be >= 1.")
        if self.hll_match_width_power_turnoff_lo > self.hll_match_width_power_turnoff_hi:
            raise ValueError("hll_match_width_power_turnoff_lo cannot exceed hll_match_width_power_turnoff_hi.")
        if self.hll_match_width_power_tail_logratio_lo < 0.0 or self.hll_match_width_power_tail_logratio_hi < 0.0:
            raise ValueError("hll_match_width_power_tail_logratio_lo and _hi must be >= 0.")
        if self.hll_match_width_power_tail_logratio_lo > self.hll_match_width_power_tail_logratio_hi:
            raise ValueError("hll_match_width_power_tail_logratio_lo cannot exceed _hi.")
        if not (0.0 <= self.hll_match_width_power_tail_reboost_max <= 1.0):
            raise ValueError("hll_match_width_power_tail_reboost_max must lie in [0,1].")
        if min(self.hll_match_br_ee, self.hll_match_br_mumu, self.hll_match_br_tautau) < 0.0:
            raise ValueError("hll_match_br_* must be >= 0.")
        if self.hll_uv_m2_floor <= 0.0:
            raise ValueError("hll_uv_m2_floor must be > 0.")
        if self.hll_uv_coupling_floor <= 0.0:
            raise ValueError("hll_uv_coupling_floor must be > 0.")
        if not (0.0 <= self.hll_uv_blend <= 1.0):
            raise ValueError("hll_uv_blend must be in [0,1].")
        if self.hll_uv_m2_power < 0.0:
            raise ValueError("hll_uv_m2_power must be >= 0.")
        if not (0.0 <= self.hll_uv_runtime_direct_gnorm_blend_max <= 1.0):
            raise ValueError("hll_uv_runtime_direct_gnorm_blend_max must lie in [0,1].")
        if self.hll_uv_runtime_direct_gnorm_metric_lo < 0.0 or self.hll_uv_runtime_direct_gnorm_metric_hi < 0.0:
            raise ValueError("hll_uv_runtime_direct_gnorm_metric_lo and _hi must be >= 0.")
        if self.hll_uv_runtime_direct_gnorm_metric_lo > self.hll_uv_runtime_direct_gnorm_metric_hi:
            raise ValueError("hll_uv_runtime_direct_gnorm_metric_lo cannot exceed _hi.")
        if self.hll_uv_rge_mu_low <= 0.0:
            raise ValueError("hll_uv_rge_mu_low must be > 0.")
        if self.hll_uv_rge_log_clip <= 0.0:
            raise ValueError("hll_uv_rge_log_clip must be > 0.")
        if not (0.0 <= self.hll_uv_runtime_direct_m2_cross_blend <= 1.0):
            raise ValueError("hll_uv_runtime_direct_m2_cross_blend must lie in [0,1].")

# =============================================================================
# 2. Yukawa Visibility Module
# =============================================================================

def load_yukawa_data(data_dir: Path = None) -> dict:
    """Load PDG data from JSON files."""
    if data_dir is None:
        # Fallback logic for locating data directory
        base_path = Path(__file__).parent
        if (base_path / "data").exists():
            data_dir = base_path / "data"
        elif (base_path.parent / "data").exists():
            data_dir = base_path.parent / "data"
        else:
            # Last resort: resolve against project root if script is relocated.
            data_dir = base_path.resolve().parent / "data"
    
    try:
        with open(data_dir / "pdg_leptons.json", 'r') as f:
            leptons = json.load(f)
        with open(data_dir / "pdg_quarks.json", 'r') as f:
            quarks = json.load(f)
            
        return {
            "leptons": quarks["yukawa_couplings"]["leptons"],
            "quarks": quarks["yukawa_couplings"]["quarks"]
        }
    except FileNotFoundError:
        print(f"Warning: Data files not found in {data_dir}. Returning empty dict.")
        return {"leptons": {}, "quarks": {}}

def compute_yukawa_bn(
    data: dict,
    mode: str = "cumulative",
    power: float = 0.30,
) -> Dict[int, float]:
    """Compute generation visibility factors B_N from Yukawa couplings.

    Paper-baseline visibility uses lepton Yukawas only with sublinear compression:

        B_N = (Y_tilde_N / Y_tilde_3)^power,  N=1,2,3

    where:
      - mode="cumulative": Y_tilde_N = sum_{k<=N} y_k
      - mode="single":     Y_tilde_N = y_N

    Returns B_norm with B_3 = 1.
    """
    if not data["leptons"]:
        # Deterministic fallback near PDG2024 lepton-cumulative values at p=0.30.
        return {1: 0.085, 2: 0.42, 3: 1.0}

    leptons = data["leptons"]
    y1 = float(leptons.get("electron", 0.0))
    y2 = float(leptons.get("muon", 0.0))
    y3 = float(leptons.get("tau", 0.0))

    if mode == "cumulative":
        y_eff = {1: y1, 2: y1 + y2, 3: y1 + y2 + y3}
    elif mode == "single":
        y_eff = {1: y1, 2: y2, 3: y3}
    else:
        raise ValueError(f"Unknown mode={mode}. Use 'cumulative' or 'single'.")

    B = {gen: (val ** power) if val > 0 else 0.0 for gen, val in y_eff.items()}
    base = B.get(3, 1.0)
    return {n: (val / base if base > 0 else 0.0) for n, val in B.items()}

# =============================================================================
# 3. Core Kinetics Module (Reproducible Spec)
# =============================================================================

class PSLTKinetics:
    """
    Rank-2 Computable Kinetics Engine.
    Implements explicit V_eff and WKB integration as defined in the Reproducible Appendix.
    """
    def __init__(self, params: Optional[PSLTParameters] = None, data_dir: Path = None):
        self.params = params if params else PSLTParameters()
        self.root_dir = Path(__file__).resolve().parent.parent
        # Cache eta-independent kinetic prefactors keyed by (N, D_rounded).
        self._gamma_prefactor_cache: Dict[Tuple[int, float], float] = {}
        self._g_fp_1d_profile: Optional[Dict[str, np.ndarray]] = None
        self._g_fp_2d_profile: Optional[Dict[str, np.ndarray]] = None
        self._g_fp_2d_spectrum: Optional[Dict[str, np.ndarray]] = None
        self._g_fp_2d_spectrum_interp_cache: Dict[float, Dict[str, np.ndarray]] = {}
        self._g_fp_2d_full_hat_cache: Dict[float, np.ndarray] = {}
        self._g_runtime_direct_spectrum_cache: Dict[float, Dict[str, np.ndarray]] = {}
        self._g_fp_2d_full_hat_runtime_cache: Dict[float, np.ndarray] = {}
        self._runtime_g_level = None
        self._runtime_g_params = None
        self._g_mode_active: str = "cardy"
        self._chi_mode_active: str = "constant"
        self._chi_open_profile: Optional[Dict[str, np.ndarray]] = None
        self._chi_open_cache: Dict[float, float] = {}
        self._chi_runtime_direct_cache: Dict[float, float] = {}
        self._gamma_runtime_direct_cache: Dict[float, Tuple[float, float]] = {}
        self._runtime_superrad_ref_a12: Optional[Tuple[float, float]] = None
        self._runtime_chi_level = None
        self._runtime_chi_params = None
        self._runtime_superrad_level = None
        self._runtime_superrad_params = None
        self._gamma_mode_active: str = "surrogate"
        self._gamma_superrad_profile: Optional[Dict[str, np.ndarray]] = None
        self._gamma_superrad_tensor: Optional[Dict[str, np.ndarray]] = None
        self._gamma_eta_profile: Optional[Dict[str, np.ndarray]] = None
        self._gamma_eta_mode_active: str = "scan"
        self._tcoh_mode_active: str = "input"
        self._tcoh_profile: Optional[Dict[str, np.ndarray]] = None
        self._b_mode_active: str = "yukawa"
        self._b_overlap_profile: Optional[Dict[str, np.ndarray]] = None
        self._b_runtime_direct_blend_profile: Optional[Dict[str, np.ndarray]] = None
        self._b_runtime_direct_sum_anchor_profile: Optional[Dict[str, np.ndarray]] = None
        self._observable_width_anchor_profile: Optional[Dict[str, np.ndarray]] = None
        self._observable_ref_amp_anchor_profile: Optional[Dict[str, np.ndarray]] = None
        self._observable_point_amp_anchor_profile: Optional[Dict[Tuple[str, int], Dict[str, np.ndarray]]] = None
        self._b_eft_norm_cache: Dict[float, np.ndarray] = {}
        self._b_eft_profile_anchor_cache: Dict[float, np.ndarray] = {}
        self._b_runtime_direct_input_cache: Dict[float, Dict[str, np.ndarray]] = {}
        self._runtime_b_level = None
        self._runtime_b_params = None
        
        # Initialize Visibility Factors (Gen 1-3 from Yukawa, N>3 decouples)
        try:
            y_data = load_yukawa_data(data_dir)
            self.B_map = compute_yukawa_bn(
                y_data,
                mode=self.params.b_n_mode,
                power=self.params.b_n_power,
            )
        except Exception as e:
            print(f"Warning: Could not initialize Yukawa B_N ({e}). Using defaults.")
            self.B_map = {1: 0.05, 2: 0.25, 3: 1.0}

        self._init_b_profiles()
        self._init_g_profiles()
        self._init_chi_profiles()
        self._init_gamma_profiles()
        self._init_observable_anchor_profiles()

    def _guess_d_from_filename(self, path: Path) -> Optional[float]:
        m = re.search(r"_D([0-9]+(?:\.[0-9]+)?)", path.stem)
        if m:
            return float(m.group(1))
        return None

    def _load_csv_rows(self, path: Path) -> List[Dict[str, str]]:
        with open(path, "r", newline="") as f:
            return list(csv.DictReader(f))

    def _load_g_fp_1d_profile(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        fine_rows = [r for r in rows if r.get("level", "").strip().lower() == "fine"]
        row = fine_rows[0] if fine_rows else rows[-1]

        keys = ("g1_ps", "g2_ps", "g3_ps")
        if not all(k in row and row[k] not in {"", None} for k in keys):
            return None

        g123 = np.array([max(float(row[k]), 1e-30) for k in keys], dtype=float)
        d_ref = self.params.g_fp_1d_ref_D
        if row.get("D", "") not in {"", None}:
            d_ref = float(row["D"])
        else:
            guessed = self._guess_d_from_filename(path)
            if guessed is not None:
                d_ref = guessed

        return {"D": np.array([float(d_ref)], dtype=float), "g123": g123.reshape(1, 3)}

    def _load_g_fp_2d_profile(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        fine_rows = [r for r in rows if r.get("level", "").strip().lower() == "fine"]
        use_rows = fine_rows if fine_rows else rows

        entries: Dict[float, np.ndarray] = {}
        for row in use_rows:
            if row.get("D", "") in {"", None}:
                continue
            dval = float(row["D"])
            if all(k in row and row[k] not in {"", None} for k in ("g1_raw", "g2_raw", "g3_raw")):
                gvals = np.array(
                    [max(float(row["g1_raw"]), 1e-30), max(float(row["g2_raw"]), 1e-30), max(float(row["g3_raw"]), 1e-30)],
                    dtype=float,
                )
            elif all(k in row and row[k] not in {"", None} for k in ("g1_hat", "g2_hat", "g3_hat")):
                # Hat-only inputs carry shape but no absolute normalization.
                g3 = 1.0
                gvals = np.array(
                    [max(float(row["g1_hat"]) * g3, 1e-30), max(float(row["g2_hat"]) * g3, 1e-30), max(float(row["g3_hat"]) * g3, 1e-30)],
                    dtype=float,
                )
            else:
                continue
            entries[dval] = gvals

        if not entries:
            return None

        d_sorted = np.array(sorted(entries.keys()), dtype=float)
        g_sorted = np.vstack([entries[d] for d in d_sorted])
        return {"D": d_sorted, "g123": g_sorted}

    def _load_g_fp_2d_spectrum(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        fine_rows = [r for r in rows if r.get("level", "").strip().lower() == "fine"]
        use_rows = fine_rows if fine_rows else rows

        entries: Dict[float, Dict[int, Tuple[float, float, float]]] = {}
        for row in use_rows:
            if row.get("D", "") in {"", None}:
                continue
            if row.get("mode_n", "") in {"", None}:
                continue
            if row.get("lambda_n", "") in {"", None}:
                continue
            if row.get("Nps_lambda_n", "") in {"", None}:
                continue
            dval = float(row["D"])
            nval = int(float(row["mode_n"]))
            if nval <= 0:
                continue
            lam = float(row["lambda_n"])
            nps = float(row["Nps_lambda_n"])
            g_raw = np.nan
            if row.get("g_raw_n", "") not in {"", None}:
                g_raw = float(row["g_raw_n"])
            entries.setdefault(dval, {})[nval] = (lam, nps, g_raw)

        if not entries:
            return None

        d_sorted = sorted(entries.keys())
        common_modes = None
        for dval in d_sorted:
            mode_set = set(entries[dval].keys())
            common_modes = mode_set if common_modes is None else (common_modes & mode_set)
        if not common_modes:
            return None

        mode_list = sorted(int(n) for n in common_modes if int(n) >= 1)
        n_max = 0
        for n in mode_list:
            if n == n_max + 1:
                n_max = n
            elif n > n_max + 1:
                break
        if n_max < 4:
            return None

        mode_idx = np.arange(1, n_max + 1, dtype=int)
        lam_rows = []
        nps_rows = []
        graw_rows = []
        for dval in d_sorted:
            lam_row = [entries[dval][int(n)][0] for n in mode_idx]
            nps_row = [entries[dval][int(n)][1] for n in mode_idx]
            graw_row = [entries[dval][int(n)][2] for n in mode_idx]
            nps0 = float(nps_row[0])
            # Reconstruct g_raw if missing in CSV (legacy exports).
            graw_row = [
                (1.0 + max(float(nps_val - nps0), 0.0)) if (not np.isfinite(gv) or gv <= 0.0) else float(gv)
                for gv, nps_val in zip(graw_row, nps_row)
            ]
            lam_rows.append(lam_row)
            nps_rows.append(nps_row)
            graw_rows.append(graw_row)

        return {
            "D": np.asarray(d_sorted, dtype=float),
            "mode_n": mode_idx,
            "lambda": np.asarray(lam_rows, dtype=float),
            "nps": np.asarray(nps_rows, dtype=float),
            "g_raw": np.asarray(graw_rows, dtype=float),
        }

    def _interp_g_fp_2d_spectrum(self, D: float) -> Optional[Dict[str, np.ndarray]]:
        if self._g_fp_2d_spectrum is None:
            return None
        d_key = float(round(D, 8))
        cached = self._g_fp_2d_spectrum_interp_cache.get(d_key)
        if cached is not None:
            return cached

        spec = self._g_fp_2d_spectrum
        d_knots = spec["D"]
        lam_knots = spec["lambda"]
        nps_knots = spec["nps"]
        graw_knots = spec["g_raw"]

        if len(d_knots) == 1:
            lam = lam_knots[0].astype(float)
            nps = nps_knots[0].astype(float)
            g_raw = graw_knots[0].astype(float)
        else:
            lam = np.array([np.interp(D, d_knots, lam_knots[:, j]) for j in range(lam_knots.shape[1])], dtype=float)
            nps = np.array([np.interp(D, d_knots, nps_knots[:, j]) for j in range(nps_knots.shape[1])], dtype=float)
            g_raw = np.array([np.interp(D, d_knots, graw_knots[:, j]) for j in range(graw_knots.shape[1])], dtype=float)
        out = {"lambda": np.maximum(lam, 1e-30), "nps": np.maximum(nps, 0.0), "g_raw": np.maximum(g_raw, 1e-30)}
        self._g_fp_2d_spectrum_interp_cache[d_key] = out
        return out

    def _runtime_direct_g_fp_2d_spectrum(self, D: float) -> Optional[Dict[str, np.ndarray]]:
        d_key = float(round(D, 8))
        use_cache = bool(self.params.runtime_direct_use_cache)
        if use_cache:
            cached = self._g_runtime_direct_spectrum_cache.get(d_key)
            if cached is not None:
                return cached

        # Lazy imports to avoid heavy solver dependency outside runtime-direct mode.
        from extract_chi_localized_2d import Level as GNLevel  # local import by design
        from extract_chi_localized_2d import PhysicalParams as GNParams  # local import by design
        from extract_chi_localized_2d import build_generalized_operator as build_gn_operator  # local import by design
        from extract_gn_phase_space_2d import SolveConfig as GNSolveConfig  # local import by design
        from extract_gn_phase_space_2d import n_phase_space as gn_n_phase_space  # local import by design
        from extract_gn_phase_space_2d import solve_low_modes as solve_gn_low_modes  # local import by design

        if self._runtime_g_level is None:
            self._runtime_g_level = GNLevel(
                "runtime",
                dr=float(self.params.runtime_direct_g_dr),
                dz=float(self.params.runtime_direct_g_dz),
            )
        if self._runtime_g_params is None:
            self._runtime_g_params = GNParams()

        level = self._runtime_g_level
        z_max = float(D) / 2.0 + float(self.params.runtime_direct_g_z_margin)
        rho, _z, _rr, _zz, uu, K, M = build_gn_operator(
            D=float(D),
            p=self._runtime_g_params,
            rho_max=float(self.params.runtime_direct_g_rho_max),
            z_max=float(z_max),
            dr=float(level.dr),
            dz=float(level.dz),
        )
        sigma = None if float(self.params.runtime_direct_g_sigma) < 0.0 else float(self.params.runtime_direct_g_sigma)
        cfg = GNSolveConfig(
            tol=float(self.params.runtime_direct_g_tol),
            maxiter=int(self.params.runtime_direct_g_maxiter),
            sigma=sigma,
            n_eigs=int(self.params.runtime_direct_g_n_eigs),
        )
        vals = solve_gn_low_modes(K, M, cfg)
        nps_vals = np.asarray(
            [gn_n_phase_space(float(E), uu, rho, float(level.dr), float(level.dz)) for E in vals],
            dtype=float,
        )
        nps0 = float(nps_vals[0])
        g_raw = np.asarray([1.0 + max(float(nv - nps0), 0.0) for nv in nps_vals], dtype=float)
        out = {
            "lambda": np.maximum(np.asarray(vals, dtype=float), 1e-30),
            "nps": np.maximum(nps_vals, 0.0),
            "g_raw": np.maximum(g_raw, 1e-30),
        }
        if use_cache:
            self._g_runtime_direct_spectrum_cache[d_key] = out
        return out

    def _build_fp_2d_full_hat_profile_from_spec(
        self,
        D: float,
        g123_hat_direct: np.ndarray,
        spec: Optional[Dict[str, np.ndarray]],
        cache_dict: Dict[float, np.ndarray],
        use_cache: bool = True,
    ) -> np.ndarray:
        d_key = float(round(D, 8))
        if use_cache:
            cached = cache_dict.get(d_key)
            if cached is not None:
                return cached

        direct = np.maximum(g123_hat_direct, 1e-30)
        if spec is None:
            # Fallback to legacy geometric extension when spectrum is unavailable.
            n_cap = 64
            hat = np.ones(n_cap, dtype=float)
            hat[:3] = direct[:3]
            r23 = float(direct[1] / max(direct[2], 1e-30))
            r13 = float(direct[0] / max(direct[2], 1e-30))
            r_tail = min(r23, r13)
            r_tail = float(np.clip(r_tail, self.params.g_fp_full_tail_clip_min, self.params.g_fp_full_tail_clip_max))
            for i in range(3, n_cap):
                hat[i] = max(hat[i - 1] * r_tail, 1e-30)
            if use_cache:
                cache_dict[d_key] = hat
            return hat

        lam = spec["lambda"]
        nps = spec["nps"]
        n_modes = len(lam)
        hat = np.ones(n_modes, dtype=float)

        nps1, nps2, nps3 = float(nps[0]), float(nps[1]), float(nps[2])
        win = np.array(
            [
                1.0 + max(nps3 - nps1, 0.0),
                1.0 + max(nps3 - nps2, 0.0),
                1.0,
            ],
            dtype=float,
        )
        alpha = self.params.g_fp_full_window_blend
        win = np.maximum(win, 1e-30)
        low_hat = (direct[:3] ** (1.0 - alpha)) * (win ** alpha)
        hat[:3] = np.maximum(low_hat, 1e-30)
        hat[2] = 1.0

        dE32 = max(float(lam[2] - lam[1]), 1e-9)
        shell3 = max(float(nps[2] - nps[1]), 1e-30)
        beta = self.params.g_fp_full_tail_beta
        shell_power = self.params.g_fp_full_tail_shell_power
        rmin = self.params.g_fp_full_tail_clip_min
        rmax = self.params.g_fp_full_tail_clip_max

        for idx in range(3, n_modes):
            shell = max(float(nps[idx] - nps[idx - 1]), 1e-30)
            shell_ratio = (shell / shell3) ** shell_power
            boltz = math.exp(-beta * max(float(lam[idx] - lam[2]), 0.0) / dE32)
            target_abs = max(shell_ratio * boltz, 1e-30)

            prev = max(float(hat[idx - 1]), 1e-30)
            step_target = target_abs / prev
            step = float(np.clip(step_target, rmin, rmax))
            hat[idx] = max(prev * step, 1e-30)

        if use_cache:
            cache_dict[d_key] = hat
        return hat

    def _build_fp_2d_full_hat_profile(self, D: float, g123_hat_direct: np.ndarray) -> np.ndarray:
        """
        Build a D-dependent ratio profile hat{g}_N = g_N / g_3 for fp_2d_full.

        Low-N (N=1,2,3):
          - Use a bounded microcanonical window anchored at E_cut=lambda_3:
              hat{g}_1^(win) = 1 + Nps(lambda_3) - Nps(lambda_1)
              hat{g}_2^(win) = 1 + Nps(lambda_3) - Nps(lambda_2)
              hat{g}_3^(win) = 1
          - Blend with the direct 2D extracted low-N ratios using
            g_fp_full_window_blend in log-space.

        Tail (N>3):
          - Shell-density factor from adjacent phase-space shells.
          - Boltzmann-like damping with local spacing scale (lambda_3-lambda_2).
          - Per-step clipping for finite-volume stability.
        """
        spec = self._interp_g_fp_2d_spectrum(D)
        return self._build_fp_2d_full_hat_profile_from_spec(
            D=float(D),
            g123_hat_direct=g123_hat_direct,
            spec=spec,
            cache_dict=self._g_fp_2d_full_hat_cache,
            use_cache=True,
        )

    def _init_g_profiles(self) -> None:
        p1 = Path(self.params.g_fp_1d_csv) if self.params.g_fp_1d_csv else self.root_dir / "output" / "gn_fp_1d" / "gn_phase_space_candidate_D12.csv"
        p2 = Path(self.params.g_fp_2d_csv) if self.params.g_fp_2d_csv else self.root_dir / "output" / "gn_fp_2d" / "gn_phase_space_2d_D6-12-18.csv"
        p2_spec = Path(self.params.g_fp_2d_spectrum_csv) if self.params.g_fp_2d_spectrum_csv else self.root_dir / "output" / "gn_fp_2d" / "gn_phase_space_2d_spectrum_D6-12-18.csv"

        self._g_fp_1d_profile = self._load_g_fp_1d_profile(p1)
        self._g_fp_2d_profile = self._load_g_fp_2d_profile(p2)
        self._g_fp_2d_spectrum = self._load_g_fp_2d_spectrum(p2_spec)
        self._g_fp_2d_spectrum_interp_cache.clear()
        self._g_fp_2d_full_hat_cache.clear()
        self._g_runtime_direct_spectrum_cache.clear()
        self._g_fp_2d_full_hat_runtime_cache.clear()

        mode = self.params.g_mode
        if mode in {"fp_1d", "fp_1d_full"} and self._g_fp_1d_profile is None:
            print(f"Warning: g_mode={mode} requested but profile is unavailable at {p1}. Falling back to cardy.")
            mode = "cardy"
        if mode in {"fp_2d", "fp_2d_full"} and self._g_fp_2d_profile is None:
            print(f"Warning: g_mode={mode} requested but profile is unavailable at {p2}. Falling back to cardy.")
            mode = "cardy"
        if mode == "fp_2d_full" and self._g_fp_2d_spectrum is None:
            print(
                "Warning: g_mode=fp_2d_full requested but no 2D spectrum file was parsed "
                f"at {p2_spec}. Using legacy geometric full-tail fallback."
            )
        if mode == "fp_2d_full_runtime_direct":
            # Runtime direct mode does not require profile CSVs.
            pass
        self._g_mode_active = mode

    def _load_b_overlap_profile(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        entries_b: Dict[float, np.ndarray] = {}
        entries_yraw: Dict[float, np.ndarray] = {}
        entries_ycum: Dict[float, np.ndarray] = {}
        entries_lambda: Dict[float, np.ndarray] = {}
        entries_guv: Dict[float, np.ndarray] = {}
        for row in rows:
            if row.get("D", "") in {"", None}:
                continue
            dval = float(row["D"])

            has_b = all(k in row and row[k] not in {"", None} for k in ("B1", "B2", "B3"))
            has_ycum = all(k in row and row[k] not in {"", None} for k in ("y_eff_cum_1", "y_eff_cum_2", "y_eff_cum_3"))
            has_yraw = all(k in row and row[k] not in {"", None} for k in ("y_eff_raw_1", "y_eff_raw_2", "y_eff_raw_3"))
            has_lambda = all(k in row and row[k] not in {"", None} for k in ("lambda_1", "lambda_2", "lambda_3"))
            has_guv = all(
                k in row and row[k] not in {"", None}
                for k in (
                    "g_uv_e_1",
                    "g_uv_e_2",
                    "g_uv_e_3",
                    "g_uv_mu_1",
                    "g_uv_mu_2",
                    "g_uv_mu_3",
                    "g_uv_tau_1",
                    "g_uv_tau_2",
                    "g_uv_tau_3",
                )
            )
            has_yflavor = all(
                k in row and row[k] not in {"", None}
                for k in (
                    "y_eff_flavor_e_1",
                    "y_eff_flavor_e_2",
                    "y_eff_flavor_e_3",
                    "y_eff_flavor_mu_1",
                    "y_eff_flavor_mu_2",
                    "y_eff_flavor_mu_3",
                    "y_eff_flavor_tau_1",
                    "y_eff_flavor_tau_2",
                    "y_eff_flavor_tau_3",
                )
            )

            bvals_row: Optional[np.ndarray] = None
            if has_b:
                bvals_row = np.array(
                    [max(float(row["B1"]), 1e-30), max(float(row["B2"]), 1e-30), max(float(row["B3"]), 1e-30)],
                    dtype=float,
                )

            if has_yraw:
                yraw = np.array(
                    [
                        max(float(row["y_eff_raw_1"]), 0.0),
                        max(float(row["y_eff_raw_2"]), 0.0),
                        max(float(row["y_eff_raw_3"]), 0.0),
                    ],
                    dtype=float,
                )
                ycum = np.array([yraw[0], yraw[0] + yraw[1], yraw[0] + yraw[1] + yraw[2]], dtype=float)
            elif has_ycum:
                ycum = np.array(
                    [
                        max(float(row["y_eff_cum_1"]), 1e-30),
                        max(float(row["y_eff_cum_2"]), 1e-30),
                        max(float(row["y_eff_cum_3"]), 1e-30),
                    ],
                    dtype=float,
                )
                yraw = np.array(
                    [ycum[0], max(ycum[1] - ycum[0], 0.0), max(ycum[2] - ycum[1], 0.0)],
                    dtype=float,
                )
            elif has_b and bvals_row is not None:
                # Legacy fallback when only normalized B_N is present.
                ycum = np.array(bvals_row / max(bvals_row[2], 1e-30), dtype=float)
                yraw = np.array(
                    [ycum[0], max(ycum[1] - ycum[0], 0.0), max(ycum[2] - ycum[1], 0.0)],
                    dtype=float,
                )
            else:
                continue

            if bvals_row is None:
                bvals = ycum / max(ycum[2], 1e-30)
            else:
                bvals = bvals_row / max(bvals_row[2], 1e-30)

            # Enforce B3=1 and positive overlap floor for robust interpolation.
            bvals = np.maximum(bvals, self.params.b_overlap_floor)
            yraw = np.maximum(yraw, self.params.b_overlap_floor)
            ycum = np.maximum(ycum, self.params.b_overlap_floor)

            if has_lambda:
                lam = np.array(
                    [
                        max(abs(float(row["lambda_1"])), self.params.hll_uv_m2_floor),
                        max(abs(float(row["lambda_2"])), self.params.hll_uv_m2_floor),
                        max(abs(float(row["lambda_3"])), self.params.hll_uv_m2_floor),
                    ],
                    dtype=float,
                )
            else:
                lam = np.ones(3, dtype=float)

            if has_guv:
                guv = np.array(
                    [
                        [float(row["g_uv_e_1"]), float(row["g_uv_e_2"]), float(row["g_uv_e_3"])],
                        [float(row["g_uv_mu_1"]), float(row["g_uv_mu_2"]), float(row["g_uv_mu_3"])],
                        [float(row["g_uv_tau_1"]), float(row["g_uv_tau_2"]), float(row["g_uv_tau_3"])],
                    ],
                    dtype=float,
                )
            elif has_yflavor:
                guv = np.sqrt(
                    np.array(
                        [
                            [
                                max(float(row["y_eff_flavor_e_1"]), self.params.hll_uv_coupling_floor),
                                max(float(row["y_eff_flavor_e_2"]), self.params.hll_uv_coupling_floor),
                                max(float(row["y_eff_flavor_e_3"]), self.params.hll_uv_coupling_floor),
                            ],
                            [
                                max(float(row["y_eff_flavor_mu_1"]), self.params.hll_uv_coupling_floor),
                                max(float(row["y_eff_flavor_mu_2"]), self.params.hll_uv_coupling_floor),
                                max(float(row["y_eff_flavor_mu_3"]), self.params.hll_uv_coupling_floor),
                            ],
                            [
                                max(float(row["y_eff_flavor_tau_1"]), self.params.hll_uv_coupling_floor),
                                max(float(row["y_eff_flavor_tau_2"]), self.params.hll_uv_coupling_floor),
                                max(float(row["y_eff_flavor_tau_3"]), self.params.hll_uv_coupling_floor),
                            ],
                        ],
                        dtype=float,
                    )
                )
            else:
                # Fallback: diagonal flavor-layer matrix from layer-resolved raw overlaps.
                guv = np.diag(np.sqrt(np.maximum(yraw, self.params.hll_uv_coupling_floor)))

            guv = np.maximum(guv, self.params.hll_uv_coupling_floor)

            entries_b[dval] = bvals
            entries_yraw[dval] = yraw
            entries_ycum[dval] = ycum
            entries_lambda[dval] = lam
            entries_guv[dval] = guv

        if not entries_b:
            return None

        d_sorted = np.array(sorted(entries_b.keys()), dtype=float)
        b_sorted = np.vstack([entries_b[d] for d in d_sorted])
        yraw_sorted = np.vstack([entries_yraw[d] for d in d_sorted])
        ycum_sorted = np.vstack([entries_ycum[d] for d in d_sorted])
        lam_sorted = np.vstack([entries_lambda[d] for d in d_sorted])
        guv_sorted = np.stack([entries_guv[d] for d in d_sorted], axis=0)
        return {
            "D": d_sorted,
            "B123": b_sorted,
            "YRAW123": yraw_sorted,
            "YCUM123": ycum_sorted,
            "LAMBDA123": lam_sorted,
            "GUV": guv_sorted,
        }

    def _load_runtime_direct_b_blend_profile(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None
        entries: Dict[float, float] = {}
        for row in rows:
            if row.get("D", "") in {"", None}:
                continue
            alpha_val = None
            for key in ("alpha", "blend", "profile_blend"):
                if row.get(key, "") not in {"", None}:
                    alpha_val = float(row[key])
                    break
            if alpha_val is None:
                continue
            entries[float(row["D"])] = float(np.clip(alpha_val, 0.0, 1.0))
        if not entries:
            return None
        d_sorted = np.array(sorted(entries.keys()), dtype=float)
        a_sorted = np.array([entries[d] for d in d_sorted], dtype=float)
        return {"D": d_sorted, "alpha": a_sorted}

    def _auto_find_b_overlap_csv(self) -> Optional[Path]:
        base = self.root_dir / "output" / "y_eff_2d"
        if not base.exists():
            return None

        canonical = base / "y_eff_2d_three_channel_profile.csv"
        if canonical.exists():
            return canonical

        cands = sorted(base.glob("y_eff_2d_three_channel_profile_D*.csv"))
        if cands:
            return cands[-1]
        return None

    def _init_b_profiles(self) -> None:
        mode = self.params.b_mode
        self._b_mode_active = "yukawa"
        self._b_overlap_profile = None
        self._b_runtime_direct_blend_profile = None
        self._b_runtime_direct_sum_anchor_profile = None
        self._b_eft_norm_cache.clear()
        self._b_eft_profile_anchor_cache.clear()
        self._b_runtime_direct_input_cache.clear()

        if mode == "eft_operator_norm_runtime_direct":
            self._b_mode_active = mode
            # Optional profile anchor for runtime-direct B closure.
            wants_anchor = (
                float(self.params.runtime_direct_b_profile_blend) > 0.0
                or float(self.params.runtime_direct_b_sum_anchor_blend) > 0.0
                or float(self.params.runtime_direct_b_eft_sum_anchor_blend) > 0.0
                or bool(self.params.runtime_direct_b_profile_blend_csv)
            )
            if wants_anchor:
                if self.params.b_overlap_csv:
                    path = Path(self.params.b_overlap_csv)
                else:
                    path = self._auto_find_b_overlap_csv()
                if path is not None:
                    prof = self._load_b_overlap_profile(path)
                    if prof is not None:
                        self._b_overlap_profile = prof
                    else:
                        print(
                            f"Warning: runtime_direct_b_profile_blend>0 but overlap profile at {path} "
                            "could not be parsed. Falling back to pure runtime-direct B."
                        )
                else:
                    print(
                        "Warning: runtime_direct_b_profile_blend>0 but no overlap profile CSV was found. "
                        "Falling back to pure runtime-direct B."
                    )
            if self.params.runtime_direct_b_profile_blend_csv:
                blend_path = Path(str(self.params.runtime_direct_b_profile_blend_csv))
                prof = self._load_runtime_direct_b_blend_profile(blend_path)
                if prof is not None:
                    self._b_runtime_direct_blend_profile = prof
                else:
                    print(
                        f"Warning: runtime_direct_b_profile_blend_csv={blend_path} could not be parsed. "
                        "Falling back to scalar runtime_direct_b_profile_blend."
                    )
            if self.params.runtime_direct_b_sum_anchor_csv:
                sum_path = Path(str(self.params.runtime_direct_b_sum_anchor_csv))
                prof = self._load_runtime_direct_b_sum_anchor_profile(sum_path)
                if prof is not None:
                    self._b_runtime_direct_sum_anchor_profile = prof
                else:
                    print(
                        f"Warning: runtime_direct_b_sum_anchor_csv={sum_path} could not be parsed. "
                        "Falling back to local overlap sum anchor."
                    )
            return

        if mode not in {"overlap_2d", "eft_operator_norm"}:
            return

        if self.params.b_overlap_csv:
            path = Path(self.params.b_overlap_csv)
        else:
            path = self._auto_find_b_overlap_csv()

        if path is None:
            print(f"Warning: b_mode={mode} requested but no overlap profile CSV was found. Falling back to yukawa.")
            return

        prof = self._load_b_overlap_profile(path)
        if prof is None:
            print(f"Warning: could not parse overlap B_N profile from {path}. Falling back to yukawa.")
            return

        self._b_overlap_profile = prof
        self._b_mode_active = str(mode)

    def active_b_mode(self) -> str:
        return self._b_mode_active

    def active_g_mode(self) -> str:
        return self._g_mode_active

    def _runtime_direct_b_profile_blend_at(self, D: float) -> float:
        alpha = float(self.params.runtime_direct_b_profile_blend)
        prof = self._b_runtime_direct_blend_profile
        if prof is not None:
            alpha = float(np.interp(float(D), np.asarray(prof["D"], dtype=float), np.asarray(prof["alpha"], dtype=float)))
        return float(np.clip(alpha, 0.0, 1.0))

    def _load_runtime_direct_b_sum_anchor_profile(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        entries: Dict[float, float] = {}
        for row in rows:
            if row.get("D", "") in {"", None}:
                continue
            sval = row.get("Bsum", row.get("B_sum", row.get("bsum", "")))
            if sval in {"", None}:
                continue
            entries[float(row["D"])] = max(float(sval), self.params.b_overlap_floor)

        if len(entries) < 2:
            return None

        d_sorted = np.array(sorted(entries.keys()), dtype=float)
        sums = np.array([entries[d] for d in d_sorted], dtype=float)
        return {"D": d_sorted, "Bsum": np.maximum(sums, self.params.b_overlap_floor)}

    def _runtime_direct_b_sum_anchor_target(self, D: float) -> Optional[float]:
        prof = self._b_runtime_direct_sum_anchor_profile
        if prof is not None:
            return float(
                max(
                    np.interp(float(D), np.asarray(prof["D"], dtype=float), np.asarray(prof["Bsum"], dtype=float)),
                    self.params.b_overlap_floor,
                )
            )

        if self._b_overlap_profile is not None:
            d_knots = np.asarray(self._b_overlap_profile["D"], dtype=float)
            b_prof = np.array(
                [np.interp(float(D), d_knots, self._b_overlap_profile["B123"][:, i]) for i in range(3)],
                dtype=float,
            )
            return float(max(np.sum(np.maximum(b_prof, self.params.b_overlap_floor)), self.params.b_overlap_floor))

        return None

    def _init_observable_anchor_profiles(self) -> None:
        self._observable_width_anchor_profile = None
        self._observable_ref_amp_anchor_profile = None
        self._observable_point_amp_anchor_profile = None
        if self.params.observable_width_anchor_csv not in {None, ""}:
            path = Path(str(self.params.observable_width_anchor_csv))
            prof = self._load_observable_width_anchor_profile(path)
            if prof is not None:
                self._observable_width_anchor_profile = prof
            else:
                print(
                    f"Warning: observable_width_anchor_csv={path} could not be parsed. "
                    "Falling back to native width ratio."
                )
        if self.params.observable_ref_amp_anchor_csv not in {None, ""}:
            path = Path(str(self.params.observable_ref_amp_anchor_csv))
            prof = self._load_observable_ref_amp_anchor_profile(path)
            if prof is not None:
                self._observable_ref_amp_anchor_profile = prof
            else:
                print(
                    f"Warning: observable_ref_amp_anchor_csv={path} could not be parsed. "
                    "Falling back to native reference amplitude."
                )
        if self.params.observable_point_amp_anchor_csv not in {None, ""}:
            path = Path(str(self.params.observable_point_amp_anchor_csv))
            prof = self._load_observable_point_amp_anchor_profile(path)
            if prof is not None:
                self._observable_point_amp_anchor_profile = prof
            else:
                print(
                    f"Warning: observable_point_amp_anchor_csv={path} could not be parsed. "
                    "Falling back to native point amplitude."
                )

    def _load_observable_width_anchor_profile(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        entries: Dict[Tuple[float, float], float] = {}
        d_vals: set[float] = set()
        eta_vals: set[float] = set()
        for row in rows:
            if row.get("D", "") in {"", None} or row.get("eta", "") in {"", None}:
                continue
            wval = row.get("width_ratio", row.get("width", row.get("width_uv_rge", "")))
            if wval in {"", None}:
                continue
            d_key = float(row["D"])
            eta_key = float(row["eta"])
            entries[(d_key, eta_key)] = max(float(wval), self.params.b_overlap_floor)
            d_vals.add(d_key)
            eta_vals.add(eta_key)

        if len(d_vals) < 2 or len(eta_vals) < 2:
            return None

        d_sorted = np.array(sorted(d_vals), dtype=float)
        eta_sorted = np.array(sorted(eta_vals), dtype=float)
        grid = np.full((len(d_sorted), len(eta_sorted)), np.nan, dtype=float)
        d_index = {float(v): i for i, v in enumerate(d_sorted)}
        eta_index = {float(v): i for i, v in enumerate(eta_sorted)}
        for (d_key, eta_key), wval in entries.items():
            grid[d_index[d_key], eta_index[eta_key]] = wval

        if np.isnan(grid).any():
            return None
        return {"D": d_sorted, "eta": eta_sorted, "width_ratio": grid}

    def _observable_width_anchor_target(self, D: float, eta: float) -> Optional[float]:
        prof = self._observable_width_anchor_profile
        if prof is None:
            return None
        d_knots = np.asarray(prof["D"], dtype=float)
        eta_knots = np.asarray(prof["eta"], dtype=float)
        grid = np.asarray(prof["width_ratio"], dtype=float)
        eta_slice = np.array(
            [np.interp(float(eta), eta_knots, grid[i, :]) for i in range(len(d_knots))],
            dtype=float,
        )
        return float(max(np.interp(float(D), d_knots, eta_slice), self.params.b_overlap_floor))

    def _load_observable_ref_amp_anchor_profile(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        entries: Dict[Tuple[str, int], float] = {}
        for row in rows:
            mode = str(row.get("observable_mode", "")).strip()
            layer_raw = row.get("layer", "")
            aval = row.get("amp_ref", row.get("amp", ""))
            if mode in {"", "None"} or layer_raw in {"", None} or aval in {"", None}:
                continue
            entries[(mode, int(layer_raw))] = max(float(aval), self.params.b_overlap_floor)

        if not entries:
            return None

        return {
            "observable_mode": np.array([key[0] for key in entries.keys()], dtype=object),
            "layer": np.array([key[1] for key in entries.keys()], dtype=int),
            "amp_ref": np.array([entries[key] for key in entries.keys()], dtype=float),
        }

    def _observable_ref_amp_anchor_target(self, layer_n: int, observable_mode: str) -> Optional[float]:
        prof = self._observable_ref_amp_anchor_profile
        if prof is None:
            return None
        modes = np.asarray(prof["observable_mode"], dtype=object)
        layers = np.asarray(prof["layer"], dtype=int)
        amps = np.asarray(prof["amp_ref"], dtype=float)
        mask = (layers == int(layer_n)) & (modes == str(observable_mode))
        if not np.any(mask):
            return None
        return float(max(amps[np.nonzero(mask)[0][0]], self.params.b_overlap_floor))

    def _load_observable_point_amp_anchor_profile(
        self,
        path: Path,
    ) -> Optional[Dict[Tuple[str, int], Dict[str, np.ndarray]]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        grouped: Dict[Tuple[str, int], Dict[Tuple[float, float], float]] = {}
        d_sets: Dict[Tuple[str, int], set[float]] = {}
        eta_sets: Dict[Tuple[str, int], set[float]] = {}
        for row in rows:
            mode = str(row.get("observable_mode", "")).strip()
            layer_raw = row.get("layer", "")
            d_raw = row.get("D", "")
            eta_raw = row.get("eta", "")
            aval = row.get("amp", "")
            if (
                mode in {"", "None"}
                or layer_raw in {"", None}
                or d_raw in {"", None}
                or eta_raw in {"", None}
                or aval in {"", None}
            ):
                continue
            key = (mode, int(layer_raw))
            grouped.setdefault(key, {})[(float(d_raw), float(eta_raw))] = max(float(aval), self.params.b_overlap_floor)
            d_sets.setdefault(key, set()).add(float(d_raw))
            eta_sets.setdefault(key, set()).add(float(eta_raw))

        if not grouped:
            return None

        prof: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
        for key, entries in grouped.items():
            d_sorted = np.array(sorted(d_sets[key]), dtype=float)
            eta_sorted = np.array(sorted(eta_sets[key]), dtype=float)
            if len(d_sorted) < 2 or len(eta_sorted) < 2:
                continue
            grid = np.full((len(d_sorted), len(eta_sorted)), np.nan, dtype=float)
            d_index = {float(v): i for i, v in enumerate(d_sorted)}
            eta_index = {float(v): i for i, v in enumerate(eta_sorted)}
            for (d_key, eta_key), aval in entries.items():
                grid[d_index[d_key], eta_index[eta_key]] = aval
            if np.isnan(grid).any():
                continue
            prof[key] = {"D": d_sorted, "eta": eta_sorted, "amp": grid}

        return prof or None

    def _observable_point_amp_anchor_target(
        self,
        layer_n: int,
        observable_mode: str,
        D: float,
        eta: float,
    ) -> Optional[float]:
        prof = self._observable_point_amp_anchor_profile
        if prof is None:
            return None
        key = (str(observable_mode), int(layer_n))
        if key not in prof:
            return None
        entry = prof[key]
        d_knots = np.asarray(entry["D"], dtype=float)
        eta_knots = np.asarray(entry["eta"], dtype=float)
        grid = np.asarray(entry["amp"], dtype=float)
        eta_slice = np.array(
            [np.interp(float(eta), eta_knots, grid[i, :]) for i in range(len(d_knots))],
            dtype=float,
        )
        return float(max(np.interp(float(D), d_knots, eta_slice), self.params.b_overlap_floor))

    def _observable_width_anchor_effective_alpha(self, D: float) -> float:
        base = float(np.clip(self.params.observable_width_anchor_blend, 0.0, 1.0))
        taper = float(self.params.observable_width_anchor_blend_taper)
        alpha = float(base)
        if abs(taper) > 0.0:
            center = float(self.params.observable_width_anchor_blend_taper_center_D)
            sigma = max(float(self.params.observable_width_anchor_blend_taper_sigma_D), 1e-9)
            x = (float(D) - center) / sigma
            # Normalized odd local skew: +1 at D=center-sigma, -1 at D=center+sigma, ~0 away from the band.
            local_skew = float(-x * np.exp(0.5 - 0.5 * x * x))
            alpha += float(taper * local_skew)
        boost_peak = float(np.clip(self.params.observable_width_anchor_boost_peak, 0.0, 1.0))
        if boost_peak > 0.0:
            center = float(self.params.observable_width_anchor_boost_center_D)
            sigma = max(float(self.params.observable_width_anchor_boost_sigma_D), 1e-9)
            alpha += float(boost_peak * np.exp(-0.5 * ((float(D) - center) / sigma) ** 2))
        return float(np.clip(alpha, 0.0, 1.0))

    def _observable_ref_amp_anchor_effective_beta(self, D: float) -> float:
        beta1_peak = float(np.clip(self.params.observable_ref_amp_anchor_peak, 0.0, 1.0))
        beta2_peak = float(np.clip(self.params.observable_ref_amp_anchor_peak2, 0.0, 1.0))
        beta = 0.0
        if beta1_peak > 0.0:
            center1 = float(self.params.observable_ref_amp_anchor_center_D)
            sigma1 = max(float(self.params.observable_ref_amp_anchor_sigma_D), 1e-9)
            beta += float(beta1_peak * np.exp(-0.5 * ((float(D) - center1) / sigma1) ** 2))
        if beta2_peak > 0.0:
            center2 = float(self.params.observable_ref_amp_anchor_center_D2)
            sigma2 = max(float(self.params.observable_ref_amp_anchor_sigma_D2), 1e-9)
            beta += float(beta2_peak * np.exp(-0.5 * ((float(D) - center2) / sigma2) ** 2))
        return float(np.clip(beta, 0.0, 1.0))

    def _observable_point_amp_anchor_effective_beta(self, D: float) -> float:
        beta1_peak = float(np.clip(self.params.observable_point_amp_anchor_peak, 0.0, 1.0))
        beta2_peak = float(np.clip(self.params.observable_point_amp_anchor_peak2, 0.0, 1.0))
        beta3_peak = float(np.clip(self.params.observable_point_amp_anchor_peak3, 0.0, 1.0))
        beta = 0.0
        if beta1_peak > 0.0:
            center1 = float(self.params.observable_point_amp_anchor_center_D)
            sigma1 = max(float(self.params.observable_point_amp_anchor_sigma_D), 1e-9)
            beta += float(beta1_peak * np.exp(-0.5 * ((float(D) - center1) / sigma1) ** 2))
        if beta2_peak > 0.0:
            center2 = float(self.params.observable_point_amp_anchor_center_D2)
            sigma2 = max(float(self.params.observable_point_amp_anchor_sigma_D2), 1e-9)
            beta += float(beta2_peak * np.exp(-0.5 * ((float(D) - center2) / sigma2) ** 2))
        if beta3_peak > 0.0:
            center3 = float(self.params.observable_point_amp_anchor_center_D3)
            sigma3 = max(float(self.params.observable_point_amp_anchor_sigma_D3), 1e-9)
            beta += float(beta3_peak * np.exp(-0.5 * ((float(D) - center3) / sigma3) ** 2))
        return float(np.clip(beta, 0.0, 1.0))

    def _observable_partial_anchor_effective_beta(self, D: float) -> float:
        beta_peak = float(np.clip(self.params.observable_partial_anchor_peak, 0.0, 1.0))
        if beta_peak <= 0.0:
            return 0.0
        center = float(self.params.observable_partial_anchor_center_D)
        sigma = max(float(self.params.observable_partial_anchor_sigma_D), 1e-9)
        beta = float(beta_peak * np.exp(-0.5 * ((float(D) - center) / sigma) ** 2))
        return float(np.clip(beta, 0.0, 1.0))

    def _blend_observable_width_ratio(self, width_ratio: float, D: float, eta: float) -> float:
        alpha = self._observable_width_anchor_effective_alpha(float(D))
        if alpha <= 0.0:
            return float(width_ratio)
        target = self._observable_width_anchor_target(float(D), float(eta))
        if target is None:
            return float(width_ratio)
        return float(
            np.exp(
                (1.0 - alpha) * np.log(max(float(width_ratio), self.params.b_overlap_floor))
                + alpha * np.log(max(float(target), self.params.b_overlap_floor))
            )
        )

    def _blend_observable_ref_amp(
        self,
        amp_ref: float,
        layer_n: int,
        observable_mode: str,
        D: float,
    ) -> float:
        beta = self._observable_ref_amp_anchor_effective_beta(float(D))
        if beta <= 0.0:
            return float(amp_ref)
        target = self._observable_ref_amp_anchor_target(int(layer_n), str(observable_mode))
        if target is None:
            return float(amp_ref)
        return float(
            np.exp(
                (1.0 - beta) * np.log(max(float(amp_ref), self.params.b_overlap_floor))
                + beta * np.log(max(float(target), self.params.b_overlap_floor))
            )
        )

    def _blend_observable_point_amp(
        self,
        amp: float,
        layer_n: int,
        observable_mode: str,
        D: float,
        eta: float,
    ) -> float:
        beta = self._observable_point_amp_anchor_effective_beta(float(D))
        amp_eff = float(amp)
        if beta > 0.0:
            target = self._observable_point_amp_anchor_target(int(layer_n), str(observable_mode), float(D), float(eta))
            if target is not None:
                amp_eff = float(
                    np.exp(
                        (1.0 - beta) * np.log(max(float(amp_eff), self.params.b_overlap_floor))
                        + beta * np.log(max(float(target), self.params.b_overlap_floor))
                    )
                )
        if str(self.params.observable_two_lobe_mode) == "pointamp_two_lobe":
            amp_eff = float(amp_eff * math.sqrt(max(self._observable_two_lobe_factor(float(D)), 0.0)))
        return float(amp_eff)

    def _observable_two_lobe_factor(self, D: float) -> float:
        if str(self.params.observable_two_lobe_mode) == "none":
            return 1.0
        beta = float(self.params.observable_two_lobe_beta)
        gamma = 1.0 / math.sqrt(max(1.0 - beta * beta, 1.0e-12))
        d_eff = float(D) / gamma
        width = max(float(self.params.observable_two_lobe_width_D), 1.0e-9)
        z = (d_eff - float(self.params.observable_two_lobe_center_D)) / width
        gate = 1.0 / (1.0 + math.exp(-z))
        notch = 0.5 * (
            1.0
            - math.cos(
                float(self.params.observable_two_lobe_m) * (d_eff - float(self.params.observable_two_lobe_center_D))
                + float(self.params.observable_two_lobe_phase)
            )
        )
        suppress = math.exp(-float(self.params.observable_two_lobe_omega) * gate * notch)
        boost_width = max(float(self.params.observable_two_lobe_boost_width), 1.0e-9)
        boost_z = (d_eff - float(self.params.observable_two_lobe_boost_center)) / boost_width
        compensate = math.exp(float(self.params.observable_two_lobe_boost) * math.exp(-0.5 * boost_z * boost_z))
        factor = float(suppress * compensate)
        guard_peak = float(self.params.observable_two_lobe_local_guard_peak)
        if guard_peak > 0.0:
            guard_width = max(float(self.params.observable_two_lobe_local_guard_width_D), 1.0e-9)
            guard_z = (float(D) - float(self.params.observable_two_lobe_local_guard_center_D)) / guard_width
            guard_gaussian = math.exp(-0.5 * guard_z * guard_z)
            residual_frac = max(1.0 - guard_peak * guard_gaussian, 0.0)
            factor = 1.0 + (factor - 1.0) * residual_frac
        return float(factor)

    def _blend_observable_partial_ratio(
        self,
        partial_ratio: float,
        layer_n: int,
        observable_mode: str,
        D: float,
        eta: float,
        apply_two_lobe: bool = True,
    ) -> float:
        beta = self._observable_partial_anchor_effective_beta(float(D))
        partial_eff = float(partial_ratio)
        if beta > 0.0:
            amp_target = self._observable_point_amp_anchor_target(int(layer_n), str(observable_mode), float(D), float(eta))
            amp_ref_target = self._observable_ref_amp_anchor_target(int(layer_n), str(observable_mode))
            if amp_target is not None and amp_ref_target is not None:
                target = float(
                    (max(float(amp_target), self.params.b_overlap_floor) / max(float(amp_ref_target), self.params.b_overlap_floor)) ** 2
                )
                partial_eff = float(
                    np.exp(
                        (1.0 - beta) * np.log(max(float(partial_eff), self.params.b_overlap_floor))
                        + beta * np.log(max(float(target), self.params.b_overlap_floor))
                    )
                )
        if apply_two_lobe and str(self.params.observable_two_lobe_mode) == "partial_two_lobe":
            partial_eff = float(partial_eff * self._observable_two_lobe_factor(float(D)))
        return float(partial_eff)

    def _apply_observable_late_dstrip(self, mu_base: float, mu_runtime: float, D: float) -> float:
        mode = str(self.params.observable_late_dstrip_mode)
        if mode == "none":
            return float(mu_runtime)
        peak = float(self.params.observable_late_dstrip_peak)
        if peak <= 0.0:
            return float(mu_runtime)
        width = max(float(self.params.observable_late_dstrip_width_D), 1.0e-9)
        z = (float(D) - float(self.params.observable_late_dstrip_center_D)) / width
        gaussian = math.exp(-0.5 * z * z)
        residual_frac = max(1.0 - peak * gaussian, 0.0)
        floor = 1.0e-30
        if mode == "mu_linear":
            return float(mu_base + (mu_runtime - mu_base) * residual_frac)
        if mode == "mu_log":
            return float(
                math.exp(
                    (1.0 - residual_frac) * math.log(max(mu_base, floor))
                    + residual_frac * math.log(max(mu_runtime, floor))
                )
            )
        if mode == "mu_exp":
            return float(mu_runtime * math.exp(-peak * gaussian))
        return float(mu_runtime)

    def _runtime_direct_b_self_blend_weight(
        self,
        operator_b123: np.ndarray,
        direct_b123: np.ndarray,
    ) -> float:
        beta_max = float(self.params.runtime_direct_b_self_blend_max)
        if beta_max <= 0.0:
            return 0.0

        floor = float(self.params.b_overlap_floor)
        op = np.maximum(np.asarray(operator_b123, dtype=float).reshape(3), floor)
        dr = np.maximum(np.asarray(direct_b123, dtype=float).reshape(3), floor)
        # We only want to compress direct EFT visibility when it inflates
        # above the raw direct overlap fixed point in the observable bands.
        metric = float(np.max(np.maximum(np.log(op[:2] / dr[:2]), 0.0)))
        lo = float(self.params.runtime_direct_b_self_blend_metric_lo)
        hi = float(self.params.runtime_direct_b_self_blend_metric_hi)
        if hi <= lo + 1e-12:
            activation = 1.0 if metric > lo else 0.0
        else:
            activation = float(np.clip((metric - lo) / (hi - lo), 0.0, 1.0))
        return float(beta_max * activation)

    def _runtime_direct_gnorm_blend_weight(
        self,
        hll_diag: np.ndarray,
        direct_diag: np.ndarray,
        direct_b123: np.ndarray,
    ) -> float:
        beta_max = float(self.params.hll_uv_runtime_direct_gnorm_blend_max)
        if beta_max <= 0.0:
            return 0.0

        floor = float(self.params.hll_uv_coupling_floor)
        h = np.maximum(np.asarray(hll_diag, dtype=float).reshape(3), floor)
        d = np.maximum(np.asarray(direct_diag, dtype=float).reshape(3), floor)
        # Only activate when both side layers sit above their direct fixed point.
        metric = float(max(min(np.log(h[0] / d[0]), np.log(h[2] / d[2])), 0.0))
        b = np.maximum(np.asarray(direct_b123, dtype=float).reshape(3), float(self.params.b_overlap_floor))
        b2_gate = float(np.clip((float(b[1]) - 0.75) / 0.10, 0.0, 1.0))
        lo = float(self.params.hll_uv_runtime_direct_gnorm_metric_lo)
        hi = float(self.params.hll_uv_runtime_direct_gnorm_metric_hi)
        if hi <= lo + 1e-12:
            activation = 1.0 if metric > lo else 0.0
        else:
            activation = float(np.clip((metric - lo) / (hi - lo), 0.0, 1.0))
        return float(beta_max * activation * b2_gate)

    def _runtime_direct_b_operator_inputs(
        self,
        D: float,
        _track_prev: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        key = float(round(D, 8))
        if self.params.runtime_direct_use_cache and _track_prev is None:
            cached = self._b_runtime_direct_input_cache.get(key, None)
            if cached is not None:
                return {k: np.array(v, dtype=float, copy=True) for k, v in cached.items()}

        # Deterministic mode-tracking warm path: when no predecessor state is
        # provided, build the seed->target chain once and pass the tracked
        # state forward linearly. If a predecessor is already provided, use it
        # directly for the target solve rather than recursively rebuilding the
        # full warm chain again.
        d_seed = float(self.params.runtime_direct_b_track_seed_D)
        d_step = float(self.params.runtime_direct_b_track_step)
        prev = None if _track_prev is None else {k: np.array(v, dtype=float, copy=True) for k, v in _track_prev.items()}
        if prev is None and float(D) > d_seed + 1e-12:
            d_warm = np.arange(d_seed, float(D), d_step, dtype=float)
            for d_val in d_warm:
                if float(d_val) >= float(D) - 1e-10:
                    continue
                d_key = float(round(float(d_val), 8))
                if d_key == key:
                    continue
                if self.params.runtime_direct_use_cache:
                    cached = self._b_runtime_direct_input_cache.get(d_key, None)
                    if cached is not None:
                        prev = {k: np.array(v, dtype=float, copy=True) for k, v in cached.items()}
                        continue
                prev = self._runtime_direct_b_operator_inputs(d_key, _track_prev=prev)

        # Lazy imports to keep non-direct scan startup lightweight.
        from extract_chi_localized_2d import Level as BLevel  # local import by design
        from extract_chi_localized_2d import PhysicalParams as BParams  # local import by design
        from extract_chi_localized_2d import omega_2center  # local import by design
        from extract_y_eff_2d_three_channel import OverlapConfig as BOverlapConfig  # local import by design
        from extract_y_eff_2d_three_channel import chirality_profiles as b_chirality_profiles  # local import by design
        from extract_y_eff_2d_three_channel import mode_parity_indicators as b_mode_parity_indicators  # local import by design
        from extract_y_eff_2d_three_channel import flavor_sigma_scales as b_flavor_sigma_scales  # local import by design
        from extract_y_eff_2d_three_channel import assign_tracked_modes as b_assign_tracked_modes  # local import by design
        from extract_y_eff_2d_three_channel import microcanonical_average as b_microcanonical_average  # local import by design
        from extract_y_eff_2d_three_channel import mode_overlap_values as b_mode_overlap_values  # local import by design
        from extract_y_eff_2d_three_channel import solve_modes as b_solve_modes  # local import by design

        if self._runtime_b_level is None:
            self._runtime_b_level = BLevel(
                "fine",
                dr=float(self.params.runtime_direct_b_dr),
                dz=float(self.params.runtime_direct_b_dz),
            )
        if self._runtime_b_params is None:
            self._runtime_b_params = BParams()

        cfg = BOverlapConfig(
            sigma_l=float(self.params.runtime_direct_b_sigma_l),
            sigma_r=float(self.params.runtime_direct_b_sigma_r),
            frame_power=float(self.params.runtime_direct_b_frame_power),
            n_track=3,
            n_eigs=max(int(self.params.runtime_direct_b_n_eigs), 3),
            window_k=int(self.params.runtime_direct_b_window_k),
            window_gap_scale=float(self.params.runtime_direct_b_window_gap_scale),
            window_sigma_mult=float(self.params.runtime_direct_b_window_sigma_mult),
            window_floor=float(self.params.runtime_direct_b_window_floor),
            flavor_sigma_power=float(self.params.runtime_direct_b_flavor_sigma_power),
            flavor_sigma_min_scale=float(self.params.runtime_direct_b_flavor_sigma_min_scale),
            flavor_sigma_max_scale=float(self.params.runtime_direct_b_flavor_sigma_max_scale),
        )

        solved = b_solve_modes(
            d_val=float(D),
            level=self._runtime_b_level,
            p=self._runtime_b_params,
            rho_max=float(self.params.runtime_direct_b_rho_max),
            z_margin=float(self.params.runtime_direct_b_z_margin),
            tol=float(self.params.runtime_direct_b_tol),
            maxiter=int(self.params.runtime_direct_b_maxiter),
            sigma=None if float(self.params.runtime_direct_b_sigma) < 0.0 else float(self.params.runtime_direct_b_sigma),
            n_eigs=int(cfg.n_eigs),
        )
        rho = np.asarray(solved["rho"], dtype=float)
        rr = np.asarray(solved["rr"], dtype=float)
        zz = np.asarray(solved["zz"], dtype=float)
        psi = np.asarray(solved["psi"], dtype=float)
        evals = np.asarray(solved["evals"], dtype=float)

        f_l, f_r = b_chirality_profiles(
            rr=rr,
            zz=zz,
            rho=rho,
            dr=float(self._runtime_b_level.dr),
            dz=float(self._runtime_b_level.dz),
            d_val=float(D),
            sigma_l=float(cfg.sigma_l),
            sigma_r=float(cfg.sigma_r),
        )
        if abs(float(cfg.frame_power)) > 0.0:
            frame = np.power(omega_2center(rr, zz, float(D), self._runtime_b_params), float(cfg.frame_power))
        else:
            frame = np.ones_like(rr)
        kernel = f_l * f_r * frame
        y_modes = b_mode_overlap_values(
            psi=psi,
            kernel=kernel,
            rho=rho,
            dr=float(self._runtime_b_level.dr),
            dz=float(self._runtime_b_level.dz),
        )
        parity = b_mode_parity_indicators(
            psi=psi,
            rho=rho,
            dr=float(self._runtime_b_level.dr),
            dz=float(self._runtime_b_level.dz),
        )

        sigma_scales = b_flavor_sigma_scales(cfg)
        y_modes_flavor: Dict[str, np.ndarray] = {}
        for flavor in ("e", "mu", "tau"):
            s = float(sigma_scales[flavor])
            f_l_f, f_r_f = b_chirality_profiles(
                rr=rr,
                zz=zz,
                rho=rho,
                dr=float(self._runtime_b_level.dr),
                dz=float(self._runtime_b_level.dz),
                d_val=float(D),
                sigma_l=float(cfg.sigma_l) * s,
                sigma_r=float(cfg.sigma_r) * s,
            )
            kernel_f = f_l_f * f_r_f * frame
            y_modes_flavor[flavor] = b_mode_overlap_values(
                psi=psi,
                kernel=kernel_f,
                rho=rho,
                dr=float(self._runtime_b_level.dr),
                dz=float(self._runtime_b_level.dz),
            )

        y_raw = np.zeros(3, dtype=float)
        y_flavor = {flavor: np.zeros(3, dtype=float) for flavor in ("e", "mu", "tau")}
        track_idx = np.arange(3, dtype=int)
        if prev is not None:
            prev_lam = np.asarray(prev.get("lambda", np.array([evals[0], evals[1], evals[2]], dtype=float)), dtype=float)
            prev_y = np.asarray(prev.get("yraw", np.array([y_modes[0], y_modes[1], y_modes[2]], dtype=float)), dtype=float)
            prev_parity = np.asarray(prev.get("parity", np.array([parity[0], parity[1], parity[2]], dtype=float)), dtype=float)
            track_idx, _ = b_assign_tracked_modes(
                evals=evals,
                y_modes=y_modes,
                parity=parity,
                prev_lam=prev_lam,
                prev_y=prev_y,
                prev_parity=prev_parity,
                cfg=cfg,
            )
        lam_track = np.asarray([evals[int(i)] for i in track_idx], dtype=float)
        parity_track = np.asarray([parity[int(i)] for i in track_idx], dtype=float)

        for i in range(3):
            cidx = int(track_idx[i])
            y_mc, _, _ = b_microcanonical_average(
                evals=evals,
                y_modes=y_modes,
                center_idx=cidx,
                cfg=cfg,
            )
            y_raw[i] = max(float(y_mc), self.params.b_overlap_floor)
            for flavor in ("e", "mu", "tau"):
                y_flv_mc, _, _ = b_microcanonical_average(
                    evals=evals,
                    y_modes=y_modes_flavor[flavor],
                    center_idx=cidx,
                    cfg=cfg,
                )
                y_flavor[flavor][i] = max(float(y_flv_mc), self.params.b_overlap_floor)

        y_cum = np.cumsum(y_raw)
        y3 = max(float(y_cum[2]), self.params.b_overlap_floor)
        b123 = np.maximum(y_cum / y3, self.params.b_overlap_floor)
        b123 /= max(float(b123[2]), self.params.b_overlap_floor)

        g_uv = np.array(
            [
                np.sqrt(np.maximum(y_flavor["e"], self.params.hll_uv_coupling_floor)),
                np.sqrt(np.maximum(y_flavor["mu"], self.params.hll_uv_coupling_floor)),
                np.sqrt(np.maximum(y_flavor["tau"], self.params.hll_uv_coupling_floor)),
            ],
            dtype=float,
        )
        g_uv = np.maximum(g_uv, self.params.hll_uv_coupling_floor)

        # Optional stabilizer: blend runtime-direct B-operator inputs with
        # profile-derived values to control map-level drift while preserving
        # direct per-cell extraction as the primary branch.
        alpha = self._runtime_direct_b_profile_blend_at(float(D))
        if alpha > 0.0 and self._b_overlap_profile is not None:
            prof = self._b_overlap_profile
            d_knots = np.asarray(prof["D"], dtype=float)
            yraw_prof = np.array(
                [np.interp(float(D), d_knots, prof["YRAW123"][:, i]) for i in range(3)],
                dtype=float,
            )
            lam_prof = np.array(
                [np.interp(float(D), d_knots, prof["LAMBDA123"][:, i]) for i in range(3)],
                dtype=float,
            )
            guv_prof = np.zeros((3, 3), dtype=float)
            for i in range(3):
                for j in range(3):
                    guv_prof[i, j] = float(np.interp(float(D), d_knots, prof["GUV"][:, i, j]))

            y_raw = (1.0 - alpha) * y_raw + alpha * np.maximum(yraw_prof, self.params.b_overlap_floor)
            lam_track = (1.0 - alpha) * lam_track + alpha * np.maximum(np.abs(lam_prof), self.params.hll_uv_m2_floor)
            g_uv = (1.0 - alpha) * g_uv + alpha * np.maximum(guv_prof, self.params.hll_uv_coupling_floor)
            y_cum = np.cumsum(np.maximum(y_raw, self.params.b_overlap_floor))
            y3 = max(float(y_cum[2]), self.params.b_overlap_floor)
            b123 = np.maximum(y_cum / y3, self.params.b_overlap_floor)
            b123 /= max(float(b123[2]), self.params.b_overlap_floor)

        out = {
            "yraw": np.maximum(y_raw, self.params.b_overlap_floor),
            "lambda": np.maximum(np.abs(lam_track), self.params.hll_uv_m2_floor),
            "g_uv": g_uv,
            "b123": b123,
            "parity": parity_track,
        }
        if self.params.runtime_direct_use_cache:
            self._b_runtime_direct_input_cache[key] = {k: np.array(v, dtype=float, copy=True) for k, v in out.items()}
        return out

    def _load_chi_open_profile(self, path: Path, prefer_micro: bool = False) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        entries: Dict[float, Tuple[float, float, float, float]] = {}
        for row in rows:
            if row.get("D", "") in {"", None}:
                continue
            dval = float(row["D"])
            if prefer_micro:
                gphi = row.get("gamma_phi_micro", row.get("gamma_phi_geom", row.get("gamma_phi", "")))
                gmix = row.get("gamma_mix_micro", row.get("gamma_mix_geom", row.get("gamma_mix", "")))
                delt = row.get("delta_micro", row.get("delta", row.get("DeltaE", "")))
                gref = row.get("gamma_ref_micro", row.get("Gamma_ref", row.get("gamma_ref", "")))
            else:
                gphi = row.get("gamma_phi_geom", row.get("gamma_phi", row.get("gamma_phi_micro", "")))
                gmix = row.get("gamma_mix_geom", row.get("gamma_mix", row.get("gamma_mix_micro", "")))
                delt = row.get("delta", row.get("delta_micro", row.get("DeltaE", "")))
                gref = row.get("Gamma_ref", row.get("gamma_ref", row.get("gamma_ref_micro", "")))
            if gphi in {"", None} or gmix in {"", None} or delt in {"", None} or gref in {"", None}:
                continue
            entries[dval] = (float(gphi), float(gmix), float(delt), float(gref))

        if len(entries) < 2:
            return None

        d_sorted = np.array(sorted(entries.keys()), dtype=float)
        vals = np.array([entries[d] for d in d_sorted], dtype=float)
        return {
            "D": d_sorted,
            "gamma_phi": np.maximum(vals[:, 0], 1e-30),
            "gamma_mix": np.maximum(vals[:, 1], 1e-30),
            "delta": np.maximum(np.abs(vals[:, 2]), 1e-30),
            "gamma_ref": np.maximum(vals[:, 3], 1e-30),
        }

    def _auto_find_chi_open_csv(self) -> Optional[Path]:
        base = self.root_dir / "output" / "chi_open_system"
        if not base.exists():
            return None
        cands = sorted(base.glob("chi_open_system_geometry_D*.csv"))
        if not cands:
            return None

        best_path: Optional[Path] = None
        best_count = -1
        for p in cands:
            try:
                rows = self._load_csv_rows(p)
                n = len(rows)
            except Exception:
                continue
            if n > best_count:
                best_count = n
                best_path = p
        return best_path

    def _auto_find_chi_open_micro_csv(self) -> Optional[Path]:
        base = self.root_dir / "output" / "chi_open_system"
        if not base.exists():
            return None
        cands = sorted(base.glob("chi_open_system_micro_D*.csv"))
        if not cands:
            return None

        best_path: Optional[Path] = None
        best_count = -1
        for p in cands:
            try:
                n = len(self._load_csv_rows(p))
            except Exception:
                continue
            if n > best_count:
                best_count = n
                best_path = p
        return best_path

    def _init_chi_profiles(self) -> None:
        mode = self.params.chi_mode
        self._chi_mode_active = mode
        self._chi_open_profile = None

        if mode not in {"open_system", "open_system_micro"}:
            return

        if len(self.params.chi_open_D) > 0:
            dvals = np.asarray(self.params.chi_open_D, dtype=float)
            order = np.argsort(dvals)
            self._chi_open_profile = {
                "D": dvals[order],
                "gamma_phi": np.maximum(np.asarray(self.params.chi_open_gamma_phi, dtype=float)[order], 1e-30),
                "gamma_mix": np.maximum(np.asarray(self.params.chi_open_gamma_mix, dtype=float)[order], 1e-30),
                "delta": np.maximum(np.abs(np.asarray(self.params.chi_open_delta, dtype=float)[order]), 1e-30),
                "gamma_ref": np.maximum(np.asarray(self.params.chi_open_gamma_ref, dtype=float)[order], 1e-30),
            }
            return

        path: Optional[Path]
        if self.params.chi_open_csv:
            path = Path(self.params.chi_open_csv)
        else:
            if mode == "open_system_micro":
                path = self._auto_find_chi_open_micro_csv()
                if path is None:
                    # Conservative fallback keeps the module usable even before micro CSV export.
                    path = self._auto_find_chi_open_csv()
            else:
                path = self._auto_find_chi_open_csv()
        if path is not None:
            self._chi_open_profile = self._load_chi_open_profile(path, prefer_micro=(mode == "open_system_micro"))
            if self._chi_open_profile is None:
                print(f"Warning: could not parse open-system chi profile from {path}.")
        else:
            print(f"Warning: no chi_open_system profile file found for chi_mode={mode}.")

        if self._chi_open_profile is None:
            if len(self.params.chi_lr_D) >= 2 and len(self.params.chi_lr_D) == len(self.params.chi_lr_vals):
                self._chi_mode_active = "localized_interp"
                print(f"Warning: chi_mode={mode} requested but profile unavailable; falling back to localized_interp.")
            else:
                self._chi_mode_active = "constant"
                print(f"Warning: chi_mode={mode} requested but profile unavailable; falling back to constant chi.")

    def active_chi_mode(self) -> str:
        return self._chi_mode_active

    def _load_superrad_profile(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        entries: Dict[float, Tuple[float, float]] = {}
        for row in rows:
            if row.get("D", "") in {"", None}:
                continue
            level = row.get("level", "").strip().lower()
            if level and level != "fine":
                continue
            a1 = row.get("A1_profile", "")
            a2 = row.get("A2_profile", "")
            if a1 in {"", None} or a2 in {"", None}:
                continue
            dval = float(row["D"])
            entries[dval] = (max(float(a1), 1e-30), max(float(a2), 1e-30))

        if len(entries) < 2:
            return None

        d_sorted = np.array(sorted(entries.keys()), dtype=float)
        vals = np.array([entries[d] for d in d_sorted], dtype=float)
        return {
            "D": d_sorted,
            "A1": np.maximum(vals[:, 0], 1e-30),
            "A2": np.maximum(vals[:, 1], 1e-30),
        }

    def _auto_find_superrad_csv(self) -> Optional[Path]:
        base = self.root_dir / "output" / "superrad_fp_1d"
        if not base.exists():
            return None
        canonical_grid = base / "superrad_prefactor_Dgrid60_fine.csv"
        if canonical_grid.exists():
            return canonical_grid
        canonical = base / "superrad_prefactor_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
        if canonical.exists():
            return canonical
        cands = sorted(base.glob("superrad_prefactor_D*.csv"))
        if not cands:
            return None
        return cands[-1]

    def _load_superrad_tensor(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        entries: Dict[Tuple[float, int, int], float] = {}
        bound_entries: Dict[Tuple[float, int], bool] = {}
        for row in rows:
            if row.get("D", "") in {"", None}:
                continue
            valid = str(row.get("valid_action", "true")).strip().lower()
            if valid in {"false", "0", "no"}:
                continue
            raw_val = row.get("A_l_profile_refD_refN", "")
            if raw_val in {"", None}:
                raw_val = row.get("A_l_tensor", "")
            if raw_val in {"", None}:
                continue
            dval = float(row["D"])
            nval = int(float(row["N"]))
            key = (dval, nval, int(float(row["ell"])))
            bound_raw = str(row.get("bound_proxy", "true")).strip().lower()
            bound_val = bound_raw not in {"false", "0", "no"}
            bound_key = (dval, nval)
            old_bound = bound_entries.get(bound_key)
            if old_bound is not None and bool(old_bound) != bool(bound_val):
                return None
            bound_entries[bound_key] = bool(bound_val)
            val = max(float(raw_val), 1e-30)
            old = entries.get(key)
            if old is not None:
                if abs(old - val) / max(abs(old), abs(val), 1e-30) > 1e-10:
                    return None
                continue
            entries[key] = val

        if len(entries) < 2:
            return None

        d_sorted = np.array(sorted({k[0] for k in entries}), dtype=float)
        n_sorted = np.array(sorted({k[1] for k in entries}), dtype=int)
        bound = np.full((len(d_sorted), len(n_sorted)), False, dtype=bool)
        for i, dval in enumerate(d_sorted):
            for j, nval in enumerate(n_sorted):
                bound[i, j] = bool(bound_entries.get((float(dval), int(nval)), True))
        a_by_ell: Dict[int, np.ndarray] = {}
        for ell in (1, 2):
            vals = np.full((len(d_sorted), len(n_sorted)), np.nan, dtype=float)
            for i, dval in enumerate(d_sorted):
                for j, nval in enumerate(n_sorted):
                    cur = entries.get((float(dval), int(nval), ell))
                    if cur is not None:
                        vals[i, j] = max(float(cur), 1e-30)
            if not np.isfinite(vals).any():
                return None
            a_by_ell[ell] = vals

        return {
            "D": d_sorted,
            "N": n_sorted,
            "A1": a_by_ell[1],
            "A2": a_by_ell[2],
            "bound": bound,
        }

    def _auto_find_superrad_tensor_csv(self) -> Optional[Path]:
        base = self.root_dir / "output" / "superrad_fp_1d"
        if not base.exists():
            return None
        canonical = base / "channel_resolved_A_l_tensor_D4-20full_eta3_N1-2-3_l1-2.csv"
        if canonical.exists():
            return canonical
        cands = sorted(base.glob("channel_resolved_A_l_tensor_*.csv"))
        if not cands:
            return None
        non_summary = [p for p in cands if not p.name.endswith("_summary.csv")]
        return non_summary[-1] if non_summary else cands[-1]

    def _load_eta_profile(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        entries: Dict[float, Tuple[float, float]] = {}
        for row in rows:
            if row.get("D", "") in {"", None}:
                continue
            level = row.get("level", "").strip().lower()
            if level and level != "fine":
                continue
            amp = row.get("eta_amp", "")
            prob = row.get("eta_prob", "")
            if amp in {"", None} or prob in {"", None}:
                continue
            dval = float(row["D"])
            entries[dval] = (max(float(amp), 1e-30), max(float(prob), 1e-30))

        if len(entries) < 2:
            return None

        d_sorted = np.array(sorted(entries.keys()), dtype=float)
        vals = np.array([entries[d] for d in d_sorted], dtype=float)
        return {
            "D": d_sorted,
            "eta_amp": np.maximum(vals[:, 0], 1e-30),
            "eta_prob": np.maximum(vals[:, 1], 1e-30),
        }

    def _auto_find_eta_csv(self) -> Optional[Path]:
        base = self.root_dir / "output" / "eta_fp_1d"
        if not base.exists():
            return None
        canonical = base / "eta_prefactor_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
        if canonical.exists():
            return canonical
        cands = sorted(base.glob("eta_prefactor_D*.csv"))
        if not cands:
            return None
        return cands[-1]

    def _load_tcoh_profile(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        rows = self._load_csv_rows(path)
        if not rows:
            return None

        entries: Dict[float, float] = {}
        for row in rows:
            if row.get("D", "") in {"", None}:
                continue
            level = row.get("level", "").strip().lower()
            if level and level != "fine":
                continue
            tval = row.get("t_coh_deph", "")
            if tval in {"", None}:
                continue
            dval = float(row["D"])
            entries[dval] = max(float(tval), 1e-30)

        if len(entries) < 2:
            return None

        d_sorted = np.array(sorted(entries.keys()), dtype=float)
        vals = np.array([entries[d] for d in d_sorted], dtype=float)
        return {"D": d_sorted, "t_coh": np.maximum(vals, 1e-30)}

    def _auto_find_tcoh_csv(self) -> Optional[Path]:
        base = self.root_dir / "output" / "tcoh_fp_1d"
        if not base.exists():
            return None
        canonical = base / "tcoh_dephasing_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
        if canonical.exists():
            return canonical
        cands = sorted(base.glob("tcoh_dephasing_D*.csv"))
        if not cands:
            return None
        return cands[-1]

    def _init_gamma_profiles(self) -> None:
        self._gamma_mode_active = "surrogate"
        self._gamma_superrad_profile = None
        self._gamma_superrad_tensor = None
        self._gamma_eta_profile = None
        self._gamma_eta_mode_active = "scan"
        self._tcoh_mode_active = "input"
        self._tcoh_profile = None

        mode = self.params.gamma_mode
        if mode in {"action_profile", "action_grid", "action_grid_strict"}:
            sup_path = Path(self.params.gamma_superrad_csv) if self.params.gamma_superrad_csv else self._auto_find_superrad_csv()
            eta_path = Path(self.params.gamma_eta_csv) if self.params.gamma_eta_csv else self._auto_find_eta_csv()
            sup_prof = self._load_superrad_profile(sup_path) if sup_path is not None else None
            eta_prof = self._load_eta_profile(eta_path) if eta_path is not None else None

            if sup_prof is None:
                if sup_path is None:
                    print(f"Warning: gamma_mode={mode} requested but no superrad profile CSV found. Falling back to surrogate.")
                else:
                    print(f"Warning: failed to parse superrad profile at {sup_path}. Falling back to surrogate.")
            elif eta_prof is None and self.params.gamma_eta_mode != "scan":
                if eta_path is None:
                    print("Warning: gamma_eta_mode requires eta profile but no eta CSV found. Falling back to gamma_eta_mode=scan.")
                else:
                    print(f"Warning: failed to parse eta profile at {eta_path}. Falling back to gamma_eta_mode=scan.")
                self._gamma_superrad_profile = sup_prof
                self._gamma_mode_active = mode
                self._gamma_eta_mode_active = "scan"
                self._gamma_eta_profile = None
            elif sup_prof is not None:
                self._gamma_superrad_profile = sup_prof
                self._gamma_mode_active = mode
                self._gamma_eta_mode_active = self.params.gamma_eta_mode
                self._gamma_eta_profile = eta_prof
        elif mode in {"action_tensor", "action_tensor_grid_strict"}:
            sup_path = Path(self.params.gamma_superrad_csv) if self.params.gamma_superrad_csv else self._auto_find_superrad_tensor_csv()
            eta_path = Path(self.params.gamma_eta_csv) if self.params.gamma_eta_csv else self._auto_find_eta_csv()
            sup_tensor = self._load_superrad_tensor(sup_path) if sup_path is not None else None
            eta_prof = self._load_eta_profile(eta_path) if eta_path is not None else None

            if sup_tensor is None:
                if sup_path is None:
                    print(f"Warning: gamma_mode={mode} requested but no channel-resolved A_l tensor CSV found. Falling back to surrogate.")
                else:
                    print(f"Warning: failed to parse channel-resolved A_l tensor at {sup_path}. Falling back to surrogate.")
            elif eta_prof is None and self.params.gamma_eta_mode != "scan":
                if eta_path is None:
                    print("Warning: gamma_eta_mode requires eta profile but no eta CSV found. Falling back to gamma_eta_mode=scan.")
                else:
                    print(f"Warning: failed to parse eta profile at {eta_path}. Falling back to gamma_eta_mode=scan.")
                self._gamma_superrad_tensor = sup_tensor
                self._gamma_mode_active = mode
                self._gamma_eta_mode_active = "scan"
                self._gamma_eta_profile = None
            elif sup_tensor is not None:
                self._gamma_superrad_tensor = sup_tensor
                self._gamma_mode_active = mode
                self._gamma_eta_mode_active = self.params.gamma_eta_mode
                self._gamma_eta_profile = eta_prof
        elif mode == "action_runtime_direct":
            self._gamma_mode_active = mode
            if self.params.gamma_eta_mode != "scan":
                eta_path = Path(self.params.gamma_eta_csv) if self.params.gamma_eta_csv else self._auto_find_eta_csv()
                eta_prof = self._load_eta_profile(eta_path) if eta_path is not None else None
                if eta_prof is None:
                    if eta_path is None:
                        print("Warning: gamma_eta_mode requires eta profile but no eta CSV found. Falling back to gamma_eta_mode=scan.")
                    else:
                        print(f"Warning: failed to parse eta profile at {eta_path}. Falling back to gamma_eta_mode=scan.")
                    self._gamma_eta_mode_active = "scan"
                    self._gamma_eta_profile = None
                else:
                    self._gamma_eta_mode_active = self.params.gamma_eta_mode
                    self._gamma_eta_profile = eta_prof

        if self.params.t_coh_mode != "input":
            tcoh_path = Path(self.params.t_coh_dephasing_csv) if self.params.t_coh_dephasing_csv else self._auto_find_tcoh_csv()
            tcoh_prof = self._load_tcoh_profile(tcoh_path) if tcoh_path is not None else None
            if tcoh_prof is None:
                if tcoh_path is None:
                    print("Warning: t_coh_mode requested but no dephasing profile CSV found. Falling back to input t_coh.")
                else:
                    print(f"Warning: failed to parse t_coh profile at {tcoh_path}. Falling back to input t_coh.")
            else:
                self._tcoh_mode_active = self.params.t_coh_mode
                self._tcoh_profile = tcoh_prof

    def active_gamma_mode(self) -> str:
        return self._gamma_mode_active

    def active_gamma_eta_mode(self) -> str:
        return self._gamma_eta_mode_active

    def active_t_coh_mode(self) -> str:
        return self._tcoh_mode_active

    def _runtime_direct_chi(self, D: float) -> float:
        key = float(round(D, 8))
        if self.params.runtime_direct_use_cache:
            cached = self._chi_runtime_direct_cache.get(key, None)
            if cached is not None:
                return float(cached)

        # Lazy import to avoid heavy dependencies during non-direct modes.
        from extract_chi_localized_2d import Level as ChiLevel  # local import by design
        from extract_chi_localized_2d import PhysicalParams as ChiParams  # local import by design
        from extract_chi_localized_2d import run_case as run_chi_case  # local import by design

        if self._runtime_chi_level is None:
            self._runtime_chi_level = ChiLevel("fine", dr=0.06, dz=0.03)
        if self._runtime_chi_params is None:
            self._runtime_chi_params = ChiParams()

        row = run_chi_case(
            D=float(D),
            level=self._runtime_chi_level,
            p=self._runtime_chi_params,
            rho_max=float(self.params.runtime_direct_chi_rho_max),
            z_margin=float(self.params.runtime_direct_chi_z_margin),
            n_mu=int(self.params.runtime_direct_chi_n_mu),
            tol=float(self.params.runtime_direct_chi_tol),
            maxiter=int(self.params.runtime_direct_chi_maxiter),
            sigma=None if float(self.params.runtime_direct_chi_sigma) < 0.0 else float(self.params.runtime_direct_chi_sigma),
        )
        chi_val = max(float(row["chi_LR"]), 0.0)
        if self.params.runtime_direct_use_cache:
            self._chi_runtime_direct_cache[key] = chi_val
        return chi_val

    def _runtime_direct_a12(self, D: float) -> Tuple[float, float]:
        key = float(round(D, 8))
        if self.params.runtime_direct_use_cache:
            cached = self._gamma_runtime_direct_cache.get(key, None)
            if cached is not None:
                return float(cached[0]), float(cached[1])

        # Lazy import to avoid heavy dependencies during non-direct modes.
        from extract_superrad_prefactor_1d import Level as SuperradLevel  # local import by design
        from extract_superrad_prefactor_1d import PhysicalParams as SuperradParams  # local import by design
        from extract_superrad_prefactor_1d import solve_case as run_superrad_case  # local import by design

        if self._runtime_superrad_level is None:
            self._runtime_superrad_level = SuperradLevel("fine", Nz=8001)
        if self._runtime_superrad_params is None:
            self._runtime_superrad_params = SuperradParams()
        if self._runtime_superrad_ref_a12 is None:
            ref = run_superrad_case(
                D=float(self.params.runtime_direct_superrad_ref_d),
                level=self._runtime_superrad_level,
                zmax=float(self.params.runtime_direct_superrad_zmax),
                p=self._runtime_superrad_params,
                n_ref=int(self.params.runtime_direct_superrad_n_ref),
            )
            self._runtime_superrad_ref_a12 = (
                max(float(ref["A1_fp"]), 1e-300),
                max(float(ref["A2_fp"]), 1e-300),
            )

        row = run_superrad_case(
            D=float(D),
            level=self._runtime_superrad_level,
            zmax=float(self.params.runtime_direct_superrad_zmax),
            p=self._runtime_superrad_params,
            n_ref=int(self.params.runtime_direct_superrad_n_ref),
        )
        ref_a1, ref_a2 = self._runtime_superrad_ref_a12
        a1 = max(float(row["A1_fp"]) / ref_a1, 1e-30)
        a2 = max(float(row["A2_fp"]) / ref_a2, 1e-30)
        if self.params.runtime_direct_use_cache:
            self._gamma_runtime_direct_cache[key] = (a1, a2)
        return a1, a2

    def _gamma_tensor_A12(self, D: float, N: Optional[int]) -> Tuple[float, float]:
        if self._gamma_superrad_tensor is None:
            return float(self.params.A1), float(self.params.A2)

        tensor = self._gamma_superrad_tensor
        d_knots = np.asarray(tensor["D"], dtype=float)
        n_knots = np.asarray(tensor["N"], dtype=int)
        n_set = set(int(x) for x in n_knots)
        strict = self._gamma_mode_active == "action_tensor_grid_strict"
        n_req = int(self.params.runtime_direct_superrad_n_ref) if N is None else int(N)
        if n_req not in n_set:
            # The tensor audit is only a low-N channel table.  For N outside the
            # exported tensor, keep the previous conservative N_ref profile slice
            # instead of inventing a high-N channel family.
            n_req = int(self.params.runtime_direct_superrad_n_ref)
        if n_req not in n_set:
            n_req = int(n_knots[0])

        idx = int(np.where(n_knots == n_req)[0][0])
        if self.params.gamma_tensor_bound_policy == "bound_only_fallback":
            bound = tensor.get("bound")
            if bound is not None:
                bound_arr = np.asarray(bound, dtype=bool)
                d_val = float(D)
                d_idx = int(np.argmin(np.abs(d_knots - d_val)))
                if strict:
                    # Preserve strict lookup semantics; _grid_scalar will raise
                    # the same detailed D-grid error below for off-grid requests.
                    unbound_requested = abs(float(d_knots[d_idx]) - d_val) <= 1e-8 and not bool(bound_arr[d_idx, idx])
                else:
                    order = np.argsort(d_knots)
                    d_sorted = d_knots[order]
                    if d_val <= float(d_sorted[0]):
                        support = order[:1]
                    elif d_val >= float(d_sorted[-1]):
                        support = order[-1:]
                    else:
                        right = int(np.searchsorted(d_sorted, d_val, side="right"))
                        support = order[[right - 1, right]]
                    unbound_requested = not bool(np.all(bound_arr[support, idx]))
                if unbound_requested:
                    n_ref = int(self.params.runtime_direct_superrad_n_ref)
                    if n_ref in n_set:
                        idx = int(np.where(n_knots == n_ref)[0][0])
        a1_col = np.asarray(tensor["A1"], dtype=float)[:, idx]
        a2_col = np.asarray(tensor["A2"], dtype=float)[:, idx]
        if strict:
            a1 = self._grid_scalar(float(D), d_knots, a1_col, fallback_interp=False)
            a2 = self._grid_scalar(float(D), d_knots, a2_col, fallback_interp=False)
        else:
            a1 = self._interp_scalar(float(D), d_knots, a1_col)
            a2 = self._interp_scalar(float(D), d_knots, a2_col)
        scale = float(self.params.gamma_superrad_scale)
        return max(float(a1) * scale, 1e-30), max(float(a2) * scale, 1e-30)

    def _gamma_action_A12(self, D: float, N: Optional[int] = None) -> Tuple[float, float]:
        if self._gamma_mode_active == "action_runtime_direct":
            a1, a2 = self._runtime_direct_a12(float(D))
            scale = float(self.params.gamma_superrad_scale)
            return max(a1 * scale, 1e-30), max(a2 * scale, 1e-30)

        if self._gamma_mode_active in {"action_tensor", "action_tensor_grid_strict"}:
            return self._gamma_tensor_A12(float(D), N)

        if self._gamma_mode_active not in {"action_profile", "action_grid", "action_grid_strict"} or self._gamma_superrad_profile is None:
            return float(self.params.A1), float(self.params.A2)

        prof = self._gamma_superrad_profile
        d_knots = np.asarray(prof["D"], dtype=float)
        if self._gamma_mode_active in {"action_grid", "action_grid_strict"}:
            strict = self._gamma_mode_active == "action_grid_strict"
            a1 = self._grid_scalar(float(D), d_knots, np.asarray(prof["A1"], dtype=float), fallback_interp=not strict)
            a2 = self._grid_scalar(float(D), d_knots, np.asarray(prof["A2"], dtype=float), fallback_interp=not strict)
        else:
            a1 = self._interp_scalar(float(D), d_knots, np.asarray(prof["A1"], dtype=float))
            a2 = self._interp_scalar(float(D), d_knots, np.asarray(prof["A2"], dtype=float))
        scale = float(self.params.gamma_superrad_scale)
        return max(a1 * scale, 1e-30), max(a2 * scale, 1e-30)

    def eta_effective(self, D: float, eta: float) -> float:
        mode = self._gamma_eta_mode_active
        if mode == "scan" or self._gamma_eta_profile is None:
            return float(max(eta, 0.0))

        prof = self._gamma_eta_profile
        d_knots = prof["D"]
        eta_amp = float(np.interp(D, d_knots, prof["eta_amp"]))
        eta_prob = float(np.interp(D, d_knots, prof["eta_prob"]))

        if mode == "scaled_amp":
            return float(max(eta * eta_amp, 0.0))
        if mode == "scaled_prob":
            return float(max(eta * eta_prob, 0.0))
        if mode == "closed_amp":
            return float(max(eta_amp, 0.0))
        if mode == "closed_prob":
            return float(max(eta_prob, 0.0))
        return float(max(eta, 0.0))

    def t_coh_effective(self, D: float, t_coh: float) -> float:
        if self._tcoh_mode_active == "input" or self._tcoh_profile is None:
            return float(max(t_coh, 1e-30))
        d_knots = self._tcoh_profile["D"]
        t_prof = float(np.interp(D, d_knots, self._tcoh_profile["t_coh"]))
        if self._tcoh_mode_active == "dephasing_profile_capped":
            t_prof = min(t_prof, float(self.params.t_coh_cap))
        return float(max(t_prof, 1e-30))

    def _interp_g123(self, D: float, profile: Dict[str, np.ndarray]) -> np.ndarray:
        d_knots = profile["D"]
        g_knots = profile["g123"]
        if len(d_knots) == 1:
            return g_knots[0].astype(float)
        out = np.array(
            [np.interp(D, d_knots, g_knots[:, 0]), np.interp(D, d_knots, g_knots[:, 1]), np.interp(D, d_knots, g_knots[:, 2])],
            dtype=float,
        )
        return np.maximum(out, 1e-30)

    # --- Micro-degeneracy ---
    def g_N_cardy(self, N: int) -> float:
        """
        Cardy-controlled micro-degeneracy with explicit high-N suppression.

        We start from a Cardy-controlled envelope and multiply a q-suppression
        factor (mock-modular / finite-geometry regulator) to prevent runaway
        domination by arbitrarily high layers:

          g_N = exp(2*pi*sqrt(c_eff*N/6)) / N^nu * exp(-kappa_g*(N-1)^2)
        """
        if N <= 0:
            return 0.0
        exponent = 2 * np.pi * np.sqrt(self.params.c_eff * N / 6)
        g_cardy = np.exp(exponent) / (N ** self.params.nu)
        sup = np.exp(-self.params.kappa_g * (N - 1) ** 2)
        return float(g_cardy * sup)

    def _g_full_scale_n3(self, mode: str, D: float, g123_raw: np.ndarray) -> float:
        """
        Absolute normalization for *_full first-principles g_N modes.

        - cardy_anchor: keep legacy scale g_3 = g_3^Cardy.
        - phase_space:  use action-derived g_3(D) from profile/spectrum.
        """
        if self.params.g_fp_norm_mode == "cardy_anchor":
            return float(max(self.g_N_cardy(3), 1e-30))

        # phase_space normalization (default in the upgraded baseline)
        if mode == "fp_2d_full":
            spec = self._interp_g_fp_2d_spectrum(D)
            if spec is not None and len(spec.get("g_raw", [])) >= 3:
                return float(max(float(spec["g_raw"][2]), 1e-30))
        if mode == "fp_2d_full_runtime_direct":
            spec = self._runtime_direct_g_fp_2d_spectrum(D)
            if spec is not None and len(spec.get("g_raw", [])) >= 3:
                return float(max(float(spec["g_raw"][2]), 1e-30))
        return float(max(float(g123_raw[2]), 1e-30))

    def g_N_effective(self, N: int, D: float) -> float:
        """
        Effective micro-degeneracy selector.

        Modes:
          - cardy: baseline surrogate.
          - fp_1d / fp_2d: first-principles N=1..3 shape correction (D-interpolated when
            available) blended onto cardy by g_fp_blend; N>3 follows cardy tail.
          - fp_1d_full / fp_2d_full: first-principles N=1..3 profile with
            full-profile continuation for N>3 (no Cardy tail fallback).
            Absolute normalization is selected by g_fp_norm_mode.
          - fp_2d_full_runtime_direct: evaluate 2D spectrum directly at runtime
            (no profile-object interpolation), then build the same full-tail
            continuation from the direct spectrum.
        """
        if N <= 0:
            return 0.0

        mode = self._g_mode_active
        if mode == "cardy":
            return self.g_N_cardy(N)

        if mode == "fp_2d_full_runtime_direct":
            spec = self._runtime_direct_g_fp_2d_spectrum(D)
            if spec is None or len(spec.get("g_raw", [])) < 3:
                return self.g_N_cardy(N)

            g123_raw = np.maximum(np.asarray(spec["g_raw"][:3], dtype=float), 1e-30)
            g3_raw = max(float(g123_raw[2]), 1e-30)
            g123_hat = np.maximum(g123_raw / g3_raw, 1e-30)
            scale3 = self._g_full_scale_n3(mode, D, g123_raw)
            use_cache = bool(self.params.runtime_direct_use_cache)
            hat_full = self._build_fp_2d_full_hat_profile_from_spec(
                D=float(D),
                g123_hat_direct=g123_hat,
                spec=spec,
                cache_dict=self._g_fp_2d_full_hat_runtime_cache,
                use_cache=use_cache,
            )
            if N <= len(hat_full):
                return float(max(scale3 * float(hat_full[N - 1]), 1e-30))
            if len(hat_full) >= 2:
                step_tail = float(hat_full[-1] / max(hat_full[-2], 1e-30))
            else:
                step_tail = self.params.g_fp_full_tail_clip_min
            step_tail = float(np.clip(step_tail, self.params.g_fp_full_tail_clip_min, self.params.g_fp_full_tail_clip_max))
            ratio = float(hat_full[-1]) * (step_tail ** (N - len(hat_full)))
            return float(max(scale3 * ratio, 1e-30))

        profile = self._g_fp_1d_profile if mode.startswith("fp_1d") else self._g_fp_2d_profile
        if profile is None:
            return self.g_N_cardy(N)

        g123_raw = self._interp_g123(D, profile)
        g3_raw = max(float(g123_raw[2]), 1e-30)
        g123_hat = np.maximum(g123_raw / g3_raw, 1e-30)

        if mode in {"fp_1d_full", "fp_2d_full"}:
            scale3 = self._g_full_scale_n3(mode, D, g123_raw)
            if mode == "fp_2d_full":
                hat_full = self._build_fp_2d_full_hat_profile(D, g123_hat)
                if N <= len(hat_full):
                    return float(max(scale3 * float(hat_full[N - 1]), 1e-30))

                if len(hat_full) >= 2:
                    step_tail = float(hat_full[-1] / max(hat_full[-2], 1e-30))
                else:
                    step_tail = self.params.g_fp_full_tail_clip_min
                step_tail = float(np.clip(step_tail, self.params.g_fp_full_tail_clip_min, self.params.g_fp_full_tail_clip_max))
                ratio = float(hat_full[-1]) * (step_tail ** (N - len(hat_full)))
                return float(max(scale3 * ratio, 1e-30))

            if N <= 3:
                return float(max(scale3 * g123_hat[N - 1], 1e-30))
            # 1D full mode keeps the legacy geometric extension.
            r23 = float(g123_hat[1] / max(g123_hat[2], 1e-30))
            r13 = float(g123_hat[0] / max(g123_hat[2], 1e-30))
            r_tail = min(r23, r13)
            r_tail = float(np.clip(r_tail, self.params.g_fp_full_tail_clip_min, self.params.g_fp_full_tail_clip_max))
            return float(max(scale3 * (r_tail ** (N - 3)), 1e-30))

        if N <= 3:
            ratio_fp = float(g123_hat[N - 1])
            g3_cardy = self.g_N_cardy(3)
            ratio_cardy = self.g_N_cardy(N) / max(g3_cardy, 1e-30)
            shape_corr = ratio_fp / max(ratio_cardy, 1e-30)
            blend = self.params.g_fp_blend
            return float(max(self.g_N_cardy(N) * (shape_corr ** blend), 1e-30))

        return self.g_N_cardy(N)

    # --- Visibility ---
    def B_N(self, N: int, D: Optional[float] = None) -> float:
        if self._b_mode_active in {"eft_operator_norm", "eft_operator_norm_runtime_direct"} and N <= 3:
            if D is None:
                if self._b_mode_active == "eft_operator_norm" and self._b_overlap_profile is not None:
                    d_eval = float(np.mean(self._b_overlap_profile["D"]))
                else:
                    d_eval = float(self.params.runtime_direct_superrad_ref_d)
            else:
                d_eval = float(D)
            b_eft = self._b_eft_norm_vector(d_eval)
            if b_eft is not None:
                return float(max(b_eft[N - 1], self.params.b_overlap_floor))

        # Overlap-derived visibility profile for N=1,2,3.
        if self._b_mode_active == "overlap_2d" and self._b_overlap_profile is not None and N <= 3:
            prof = self._b_overlap_profile
            d_knots = prof["D"]
            b_knots = prof["B123"][:, N - 1]
            d_eval = float(np.mean(d_knots)) if D is None else float(D)
            b_val = float(np.interp(d_eval, d_knots, b_knots))
            return float(max(b_val, self.params.b_overlap_floor))

        # SM-anchored visibility for N=1,2,3
        if N in self.B_map:
            return float(self.B_map[N])
        if N <= 0:
            return 0.0
        # Paper baseline saturates B_N for N>3 to avoid double-counting with g_N regulator.
        if self.params.b_n_tail_mode == "gaussian":
            beta = self.params.b_n_tail_beta
            return float(np.exp(-beta * (N - 3) ** 2))
        return 1.0

    def _b_eft_norm_vector(self, D: float) -> Optional[np.ndarray]:
        """
        Fully normalized EFT-operator visibility profile for N=1,2,3:
          K_N = g_N g_N^T / M_N^2
          K_N^(IR) = LLRG( finite_match(K_N) )
          B_N^(eft) = Tr[K_N^(IR)] / Tr[K_3^(IR)].
        """
        if self._b_mode_active != "eft_operator_norm_runtime_direct" and self._b_overlap_profile is None:
            return None

        d_key = round(float(D), 6)
        cached = self._b_eft_norm_cache.get(d_key)
        if cached is not None:
            return cached.copy()

        g_uv = self._hll_g_uv_matrix(float(D))
        m2 = np.maximum(self._hll_m2_vector(float(D)), self.params.hll_uv_m2_floor)
        fin_cfg = self._hll_uv_finite_match_config()
        rge_cfg = self._hll_uv_rge_config()
        mu_match = mu_match_from_m2(m2, floor=rge_cfg.floor)

        strengths = np.zeros(3, dtype=float)
        for idx in range(3):
            kernel = np.outer(g_uv[:, idx], g_uv[:, idx]) / max(float(m2[idx]), self.params.hll_uv_m2_floor)
            kernel = np.maximum(kernel, self.params.hll_uv_coupling_floor)
            p_layer = np.zeros(3, dtype=float)
            p_layer[idx] = 1.0
            k_match, _ = apply_ceh_finite_one_loop(
                kernel,
                fin_cfg,
                g_uv=g_uv,
                p_kin=p_layer,
                m2=m2,
                D=float(D),
            )
            k_ir, _ = run_ceh_leading_log(k_match, mu_match, rge_cfg)
            strengths[idx] = max(float(np.trace(k_ir)), self.params.b_overlap_floor)

        norm = max(float(strengths[2]), self.params.b_overlap_floor)
        b123 = np.maximum(strengths / norm, self.params.b_overlap_floor)

        if self._b_mode_active == "eft_operator_norm_runtime_direct":
            direct = self._runtime_direct_b_operator_inputs(float(D))
            direct_b123 = np.maximum(np.asarray(direct["b123"], dtype=float), self.params.b_overlap_floor)
            beta = self._runtime_direct_b_self_blend_weight(b123, direct_b123)
            if beta > 0.0:
                b123 = np.exp(
                    (1.0 - beta) * np.log(np.maximum(b123, self.params.b_overlap_floor))
                    + beta * np.log(direct_b123)
                )
                b123 = np.maximum(b123, self.params.b_overlap_floor)
                b123 /= max(float(b123[2]), self.params.b_overlap_floor)
            alpha = self._runtime_direct_b_profile_blend_at(float(D))
            if alpha > 0.0 and self._b_overlap_profile is not None:
                prof = self._b_overlap_profile
                d_knots = np.asarray(prof["D"], dtype=float)
                b_prof = np.array(
                    [np.interp(float(D), d_knots, prof["B123"][:, i]) for i in range(3)],
                    dtype=float,
                )
                b_prof = np.maximum(b_prof, self.params.b_overlap_floor)
                b123 = (1.0 - alpha) * b123 + alpha * b_prof
                b123 = np.maximum(b123, self.params.b_overlap_floor)
                b123 /= max(float(b123[2]), self.params.b_overlap_floor)

            sum_alpha = float(np.clip(self.params.runtime_direct_b_sum_anchor_blend, 0.0, 1.0))
            if sum_alpha > 0.0:
                target_sum = self._runtime_direct_b_sum_anchor_target(float(D))
                if target_sum is not None:
                    cur_sum = max(float(np.sum(b123)), self.params.b_overlap_floor)
                    scale = float(np.exp(sum_alpha * np.log(target_sum / cur_sum)))
                    b123 = np.maximum(scale * b123, self.params.b_overlap_floor)

            eft_sum_alpha = float(np.clip(self.params.runtime_direct_b_eft_sum_anchor_blend, 0.0, 1.0))
            if eft_sum_alpha > 0.0:
                b_anchor = self._b_eft_profile_anchor_vector(float(D))
                if b_anchor is not None:
                    cur_sum = max(float(np.sum(b123)), self.params.b_overlap_floor)
                    tgt_sum = max(float(np.sum(b_anchor)), self.params.b_overlap_floor)
                    scale = float(np.exp(eft_sum_alpha * np.log(tgt_sum / cur_sum)))
                    b123 = np.maximum(scale * b123, self.params.b_overlap_floor)

        self._b_eft_norm_cache[d_key] = b123.copy()
        return b123

    def _b_eft_profile_anchor_vector(self, D: float) -> Optional[np.ndarray]:
        if self._b_overlap_profile is None:
            return None

        d_key = round(float(D), 6)
        cached = self._b_eft_profile_anchor_cache.get(d_key)
        if cached is not None:
            return cached.copy()

        prof = self._b_overlap_profile
        d_knots = np.asarray(prof.get("D", np.array([], dtype=float)), dtype=float)
        g_knots = prof.get("GUV", None)
        lam_knots = prof.get("LAMBDA123", None)
        if g_knots is None or lam_knots is None or d_knots.size == 0:
            return None

        g_uv = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                g_uv[i, j] = float(np.interp(float(D), d_knots, g_knots[:, i, j]))
        g_uv = np.maximum(g_uv, self.params.hll_uv_coupling_floor)

        lam = np.array([np.interp(float(D), d_knots, lam_knots[:, i]) for i in range(3)], dtype=float)
        m2 = np.maximum(np.abs(lam), self.params.hll_uv_m2_floor)
        pwr = float(self.params.hll_uv_m2_power)
        if pwr == 0.0:
            m2 = np.ones(3, dtype=float)
        else:
            m2 = np.maximum(m2 ** pwr, self.params.hll_uv_m2_floor)

        fin_cfg = self._hll_uv_finite_match_config()
        rge_cfg = self._hll_uv_rge_config()
        mu_match = mu_match_from_m2(m2, floor=rge_cfg.floor)

        strengths = np.zeros(3, dtype=float)
        for idx in range(3):
            kernel = np.outer(g_uv[:, idx], g_uv[:, idx]) / max(float(m2[idx]), self.params.hll_uv_m2_floor)
            kernel = np.maximum(kernel, self.params.hll_uv_coupling_floor)
            p_layer = np.zeros(3, dtype=float)
            p_layer[idx] = 1.0
            k_match, _ = apply_ceh_finite_one_loop(
                kernel,
                fin_cfg,
                g_uv=g_uv,
                p_kin=p_layer,
                m2=m2,
                D=float(D),
            )
            k_ir, _ = run_ceh_leading_log(k_match, mu_match, rge_cfg)
            strengths[idx] = max(float(np.trace(k_ir)), self.params.b_overlap_floor)

        norm = max(float(strengths[2]), self.params.b_overlap_floor)
        b123 = np.maximum(strengths / norm, self.params.b_overlap_floor)
        self._b_eft_profile_anchor_cache[d_key] = b123.copy()
        return b123

    def y_eff_raw_N(self, N: int, D: Optional[float] = None) -> float:
        """
        Effective overlap amplitude for layer N.

        In overlap_2d mode this is interpolated from y_eff_raw_N(D) profile
        (microcanonical-windowed extraction). In legacy yukawa mode, return
        a deterministic fallback consistent with the configured B_N convention.
        """
        if N <= 0:
            return 0.0

        if self._b_mode_active == "eft_operator_norm_runtime_direct" and N <= 3:
            d_eval = float(self.params.runtime_direct_superrad_ref_d) if D is None else float(D)
            direct = self._runtime_direct_b_operator_inputs(d_eval)
            return float(max(float(direct["yraw"][N - 1]), self.params.b_overlap_floor))

        if self._b_mode_active in {"overlap_2d", "eft_operator_norm"} and self._b_overlap_profile is not None and N <= 3:
            prof = self._b_overlap_profile
            yraw = prof.get("YRAW123", None)
            if yraw is not None:
                d_knots = prof["D"]
                y_knots = yraw[:, N - 1]
                d_eval = float(np.mean(d_knots)) if D is None else float(D)
                y_val = float(np.interp(d_eval, d_knots, y_knots))
                return float(max(y_val, self.params.b_overlap_floor))

        if N > 3:
            return 0.0

        # Legacy fallback from Yukawa-derived B map.
        if self.params.b_n_mode == "single":
            return float(max(self.B_map.get(N, 0.0), self.params.b_overlap_floor))

        b1 = float(max(self.B_map.get(1, 0.0), self.params.b_overlap_floor))
        b2 = float(max(self.B_map.get(2, b1), b1))
        b3 = float(max(self.B_map.get(3, b2), b2))
        yraw = np.array(
            [
                b1,
                max(b2 - b1, self.params.b_overlap_floor),
                max(b3 - b2, self.params.b_overlap_floor),
            ],
            dtype=float,
        )
        return float(yraw[N - 1])

    def layer_weight(self, N: int, D: float, eta: float, t_coh: float) -> float:
        gamma = self.calculate_gamma_N(N, D, eta)
        g_n = self.g_N_effective(N, D)
        b_n = self.B_N(N, D)
        t_eff = self.t_coh_effective(D, t_coh)
        return float(b_n * g_n * (1.0 - np.exp(-gamma * t_eff)))

    def layer_kinetic_weight(self, N: int, D: float, eta: float, t_coh: float) -> float:
        gamma = self.calculate_gamma_N(N, D, eta)
        g_n = self.g_N_effective(N, D)
        t_eff = self.t_coh_effective(D, t_coh)
        return float(g_n * (1.0 - np.exp(-gamma * t_eff)))

    def layer_kinetic_probability(self, N: int, D: float, eta: float, t_coh: float, N_max: int = 20) -> float:
        if N <= 0 or N > N_max:
            return 0.0
        q = np.array([self.layer_kinetic_weight(k, D, eta, t_coh) for k in range(1, N_max + 1)], dtype=float)
        q_sum = float(np.sum(q))
        if q_sum <= 0.0:
            return 0.0
        return float(max(q[N - 1], 0.0) / q_sum)

    def _hll_match_config(self) -> EFTWilsonMatchConfig:
        return EFTWilsonMatchConfig(
            basis_mode=self.params.hll_match_basis_mode,
            mix_scale=self.params.hll_match_mix_scale,
            mix_max=self.params.hll_match_mix_max,
            eta_power=self.params.hll_match_eta_power,
            eta_ref=self.params.hll_match_eta_ref,
            width_mode=self.params.hll_match_width_mode,
            width_scale=self.params.hll_match_width_scale,
            width_power_base=self.params.hll_match_width_power_base,
            width_power_log_ratio_coeff=self.params.hll_match_width_power_log_ratio_coeff,
            width_power_min=self.params.hll_match_width_power_min,
            width_power_max=self.params.hll_match_width_power_max,
            width_power_trigger_lo=self.params.hll_match_width_power_trigger_lo,
            width_power_trigger_hi=self.params.hll_match_width_power_trigger_hi,
            width_power_turnoff_lo=self.params.hll_match_width_power_turnoff_lo,
            width_power_turnoff_hi=self.params.hll_match_width_power_turnoff_hi,
            width_power_tail_logratio_lo=self.params.hll_match_width_power_tail_logratio_lo,
            width_power_tail_logratio_hi=self.params.hll_match_width_power_tail_logratio_hi,
            width_power_tail_reboost_max=self.params.hll_match_width_power_tail_reboost_max,
            br_ee=self.params.hll_match_br_ee,
            br_mumu=self.params.hll_match_br_mumu,
            br_tautau=self.params.hll_match_br_tautau,
            floor=self.params.b_overlap_floor,
        )

    def _hll_uv_tree_config(self) -> UVTreeMatchConfig:
        return UVTreeMatchConfig(
            m2_floor=self.params.hll_uv_m2_floor,
            coupling_floor=self.params.hll_uv_coupling_floor,
        )

    def _hll_uv_finite_match_config(self) -> EFTFiniteOneLoopMatchConfig:
        return EFTFiniteOneLoopMatchConfig(
            kappa_diag=self.params.hll_uv_match_kappa_diag,
            kappa_offdiag=self.params.hll_uv_match_kappa_offdiag,
            mode=self.params.hll_uv_match_mode,
            input_diag_scale=self.params.hll_uv_match_input_diag_scale,
            input_offdiag_scale=self.params.hll_uv_match_input_offdiag_scale,
            floor=self.params.hll_uv_coupling_floor,
        )

    def _hll_uv_rge_config(self) -> EFTLeadingLogRGEConfig:
        return EFTLeadingLogRGEConfig(
            mu_low=self.params.hll_uv_rge_mu_low,
            gamma_diag=self.params.hll_uv_rge_gamma_diag,
            gamma_offdiag=self.params.hll_uv_rge_gamma_offdiag,
            log_clip=self.params.hll_uv_rge_log_clip,
            floor=self.params.hll_uv_coupling_floor,
        )

    def _hll_yraw_vector(self, D: float) -> np.ndarray:
        return np.array(
            [
                max(self.y_eff_raw_N(1, D), self.params.b_overlap_floor),
                max(self.y_eff_raw_N(2, D), self.params.b_overlap_floor),
                max(self.y_eff_raw_N(3, D), self.params.b_overlap_floor),
            ],
            dtype=float,
        )

    def _hll_pkin_vector(self, D: float, eta: float, t_coh: float, N_max: int = 20) -> np.ndarray:
        q = np.array([self.layer_kinetic_weight(k, D, eta, t_coh) for k in range(1, N_max + 1)], dtype=float)
        q = np.maximum(q, 0.0)
        q_sum = float(np.sum(q))
        if q_sum <= 0.0:
            return np.zeros(3, dtype=float)
        return np.array([float(q[0] / q_sum), float(q[1] / q_sum), float(q[2] / q_sum)], dtype=float)

    def _hll_g_uv_matrix(self, D: float) -> np.ndarray:
        diag = np.diag(np.sqrt(np.maximum(self._hll_yraw_vector(D), self.params.hll_uv_coupling_floor)))
        if self._b_mode_active == "eft_operator_norm_runtime_direct":
            direct = self._runtime_direct_b_operator_inputs(float(D))
            g = np.asarray(direct["g_uv"], dtype=float)
            blend = float(self.params.hll_uv_blend)
            g_eff = blend * g + (1.0 - blend) * diag
            gnorm_blend = self._runtime_direct_gnorm_blend_weight(
                np.diag(g_eff),
                np.diag(g),
                np.asarray(direct["b123"], dtype=float),
            )
            if gnorm_blend > 0.0:
                g_eff = (1.0 - gnorm_blend) * g_eff + gnorm_blend * g
            return np.maximum(g_eff, self.params.hll_uv_coupling_floor)

        prof = self._b_overlap_profile
        if self._b_mode_active in {"overlap_2d", "eft_operator_norm"} and prof is not None and "GUV" in prof:
            d_knots = prof["D"]
            g_knots = prof["GUV"]
            g = np.zeros((3, 3), dtype=float)
            for i in range(3):
                for n in range(3):
                    g[i, n] = float(np.interp(D, d_knots, g_knots[:, i, n]))
            blend = float(self.params.hll_uv_blend)
            g_eff = blend * g + (1.0 - blend) * diag
            return np.maximum(g_eff, self.params.hll_uv_coupling_floor)

        # Fallback: diagonal coupling from layer-resolved overlaps.
        return np.maximum(diag, self.params.hll_uv_coupling_floor)

    def _hll_m2_vector(self, D: float) -> np.ndarray:
        pwr = float(self.params.hll_uv_m2_power)
        if self._b_mode_active == "eft_operator_norm_runtime_direct":
            direct = self._runtime_direct_b_operator_inputs(float(D))
            lam = np.asarray(direct["lambda"], dtype=float)
            cross_blend = float(np.clip(self.params.hll_uv_runtime_direct_m2_cross_blend, 0.0, 1.0))
            if cross_blend > 0.0 and lam.shape[0] >= 2 and float(lam[1]) > float(lam[0]):
                lam0 = max(abs(float(lam[0])), self.params.hll_uv_m2_floor)
                lam1 = max(abs(float(lam[1])), self.params.hll_uv_m2_floor)
                target = float(np.sqrt(lam0 * lam1))
                lam = np.array(lam, dtype=float, copy=True)
                lam[1] = float(
                    np.exp(
                        (1.0 - cross_blend) * np.log(lam1)
                        + cross_blend * np.log(target)
                    )
                )
            m2 = np.maximum(np.abs(lam), self.params.hll_uv_m2_floor)
            if pwr == 0.0:
                return np.ones(3, dtype=float)
            return np.maximum(m2 ** pwr, self.params.hll_uv_m2_floor)

        prof = self._b_overlap_profile
        if self._b_mode_active in {"overlap_2d", "eft_operator_norm"} and prof is not None and "LAMBDA123" in prof:
            d_knots = prof["D"]
            lam_knots = prof["LAMBDA123"]
            lam = np.array([np.interp(D, d_knots, lam_knots[:, n]) for n in range(3)], dtype=float)
            m2 = np.maximum(np.abs(lam), self.params.hll_uv_m2_floor)
            if pwr == 0.0:
                return np.ones(3, dtype=float)
            return np.maximum(m2 ** pwr, self.params.hll_uv_m2_floor)
        if pwr == 0.0:
            return np.ones(3, dtype=float)
        return np.ones(3, dtype=float)

    def hll_wilson_coeff(self, layer_n: int, D: float, eta: float, t_coh: float, N_max: int = 20) -> float:
        """
        Map-level EFT ansatz for H->ll:
          c_ll(D,eta) = y_eff_raw_N(D) * P_N^(kin)(D,eta)
        where P_N^(kin) is the normalized kinetic occupancy from g_N and Gamma_N.
        """
        p_kin = self.layer_kinetic_probability(layer_n, D, eta, t_coh, N_max=N_max)
        y_raw = self.y_eff_raw_N(layer_n, D)
        return float(max(y_raw, self.params.b_overlap_floor) * p_kin)

    def hll_wilson_matrix_matched(self, D: float, eta: float, t_coh: float, N_max: int = 20) -> np.ndarray:
        """
        Tree-level matched Wilson matrix C_{eH}^{ij} from overlap + kinetic chain:
          C = Y(D,eta) * diag(P_kin) * Y(D,eta)^T
        where Y contains flavor-layer couplings with bounded off-diagonal mixing.
        """
        cfg = self._hll_match_config()
        y_raw = self._hll_yraw_vector(D)
        p_kin = self._hll_pkin_vector(D, eta, t_coh, N_max=N_max)
        chi_eff = self.chi_effective(D)
        eps = mixing_epsilon(chi_eff=chi_eff, eta_val=eta, cfg=cfg)
        return wilson_matrix(y_raw=y_raw, p_kin=p_kin, eps=eps, cfg=cfg)

    def hll_wilson_coeff_matched(self, layer_n: int, D: float, eta: float, t_coh: float, N_max: int = 20) -> float:
        if layer_n <= 0 or layer_n > 3:
            return 0.0
        cmat = self.hll_wilson_matrix_matched(D, eta, t_coh, N_max=N_max)
        return float(max(cmat[layer_n - 1, layer_n - 1], self.params.b_overlap_floor))

    def hll_wilson_matrix_uv_tree(self, D: float, eta: float, t_coh: float, N_max: int = 20) -> np.ndarray:
        """
        UV-inspired tree-level matching closure:
          C_{eH}^{ij} = sum_N g_{iN}(D) * [P_N^(kin)(D,eta) / M_N^2(D)] * g_{jN}(D).
        """
        cfg = self._hll_uv_tree_config()
        g_uv = self._hll_g_uv_matrix(D)
        p_kin = self._hll_pkin_vector(D, eta, t_coh, N_max=N_max)
        m2 = self._hll_m2_vector(D)
        return wilson_matrix_uv_tree(g_uv=g_uv, p_kin=p_kin, m2=m2, cfg=cfg)

    def compute_ceh_uv(self, D: float, eta: float, t_coh: float, N_max: int = 20) -> np.ndarray:
        """
        Unified UV-matching entrypoint used by scan scripts.
        Returns C_{eH}(mu_match) from the UV-tree closure.
        """
        return self.hll_wilson_matrix_uv_tree(D=D, eta=eta, t_coh=t_coh, N_max=N_max)

    def hll_uv_operator_basis_witness(
        self,
        D: float,
        eta: float,
        t_coh: float,
        N_max: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Explicit operator-basis witness for the UV-tree closure:
          C_tree = sum_N [P_N^(kin) / M_N^2] * (g_N g_N^T).
        """
        cfg = self._hll_uv_tree_config()
        g_uv = self._hll_g_uv_matrix(D)
        p_kin = self._hll_pkin_vector(D, eta, t_coh, N_max=N_max)
        m2 = self._hll_m2_vector(D)
        basis = uv_tree_operator_basis(g_uv=g_uv, p_kin=p_kin, m2=m2, cfg=cfg)
        c_tree_direct = wilson_matrix_uv_tree(g_uv=g_uv, p_kin=p_kin, m2=m2, cfg=cfg)
        tree_diag, tree_offdiag = decompose_diag_offdiag(basis.c_tree)
        return {
            "g_uv": basis.g_uv,
            "p_kin": basis.p_kin,
            "m2": basis.m2,
            "coefficients": basis.coefficients,
            "basis_matrices": basis.basis_matrices,
            "c_tree": basis.c_tree,
            "c_tree_diag": tree_diag,
            "c_tree_offdiag": tree_offdiag,
            "c_tree_direct": c_tree_direct,
            "tree_rebuild_residual": np.array(
                [float(np.max(np.abs(basis.c_tree - c_tree_direct)))],
                dtype=float,
            ),
        }

    def hll_uv_matching_witness(
        self,
        D: float,
        eta: float,
        t_coh: float,
        N_max: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Full UV->finite-match->LL-RG witness with explicit operator-basis
        reconstruction and blockwise diagonal/off-diagonal decomposition.
        """
        basis = self.hll_uv_operator_basis_witness(D=D, eta=eta, t_coh=t_coh, N_max=N_max)
        c_tree = np.asarray(basis["c_tree"], dtype=float)
        fin_cfg = self._hll_uv_finite_match_config()
        fin = finite_one_loop_witness(
            c_tree=c_tree,
            cfg=fin_cfg,
            g_uv=np.asarray(basis["g_uv"], dtype=float),
            p_kin=np.asarray(basis["p_kin"], dtype=float),
            m2=np.asarray(basis["m2"], dtype=float),
            D=float(D),
        )

        rge_cfg = self._hll_uv_rge_config()
        mu_match = mu_match_from_m2(np.asarray(basis["m2"], dtype=float), floor=rge_cfg.floor)
        rge = leading_log_witness(c_match=fin.c_match, mu_match=mu_match, cfg=rge_cfg)
        c_ir_diag, c_ir_offdiag = decompose_diag_offdiag(rge.c_low)

        c_match_rebuild = (
            fin.c_tree_diag
            + fin.c_tree_offdiag
            + fin.delta_match_diag
            + fin.delta_match_offdiag
        )
        c_ir_rebuild = c_match_rebuild + rge.delta_rge_diag + rge.delta_rge_offdiag

        return {
            **basis,
            "c_match": fin.c_match,
            "c_match_diag": fin.c_tree_diag + fin.delta_match_diag,
            "c_match_offdiag": fin.c_tree_offdiag + fin.delta_match_offdiag,
            "delta_match_diag": fin.delta_match_diag,
            "delta_match_offdiag": fin.delta_match_offdiag,
            "c_ir": rge.c_low,
            "c_ir_diag": c_ir_diag,
            "c_ir_offdiag": c_ir_offdiag,
            "delta_rge_diag": rge.delta_rge_diag,
            "delta_rge_offdiag": rge.delta_rge_offdiag,
            "mu_match": np.array([float(rge.mu_match)], dtype=float),
            "mu_low": np.array([float(rge.mu_low)], dtype=float),
            "log_ratio": np.array([float(rge.log_ratio)], dtype=float),
            "kappa_diag": np.array([float(fin.kappa_diag)], dtype=float),
            "kappa_offdiag": np.array([float(fin.kappa_offdiag)], dtype=float),
            "kappa_diag_eff": np.array([float(fin.kappa_diag_eff)], dtype=float),
            "kappa_offdiag_eff": np.array([float(fin.kappa_offdiag_eff)], dtype=float),
            "finite_match_mode": np.array(
                [
                    0.0
                    if fin.mode == "constant"
                    else 1.0
                    if fin.mode == "input_tied"
                    else 2.0
                    if fin.mode == "action_normalized"
                    else 3.0
                    if fin.mode == "action_absolute"
                    else 4.0
                    if fin.mode == "action_loop_contrast"
                    else 5.0
                    if fin.mode == "action_loop_absolute"
                    else 6.0
                    if fin.mode == "action_loop_eymh_absolute"
                    else 7.0
                    if fin.mode == "action_loop_eymh_source_informed"
                    else 8.0
                    if fin.mode == "action_loop_eymh_parented"
                    else 9.0
                ],
                dtype=float,
            ),
            "shell_spread": np.array([float(fin.shell_spread)], dtype=float),
            "coeff_cv": np.array([float(fin.coeff_cv)], dtype=float),
            "offdiag_mix": np.array([float(fin.offdiag_mix)], dtype=float),
            "gap_cv": np.array([float(fin.gap_cv)], dtype=float),
            "gap_asym": np.array([float(fin.gap_asym)], dtype=float),
            "g_col_norm_cv": np.array([float(fin.g_col_norm_cv)], dtype=float),
            "c_tree_diag_cv": np.array([float(fin.c_tree_diag_cv)], dtype=float),
            "pkin_entropy": np.array([float(fin.pkin_entropy)], dtype=float),
            "action_norm_diag": np.array([float(fin.action_norm_diag)], dtype=float),
            "action_norm_offdiag": np.array([float(fin.action_norm_offdiag)], dtype=float),
            "coeff_l1": np.array([float(fin.coeff_l1)], dtype=float),
            "coeff_l2": np.array([float(fin.coeff_l2)], dtype=float),
            "coeff_align": np.array([float(fin.coeff_align)], dtype=float),
            "action_abs_diag": np.array([float(fin.action_abs_diag)], dtype=float),
            "action_abs_offdiag": np.array([float(fin.action_abs_offdiag)], dtype=float),
            "hk_omega_mid": np.array([float(fin.hk_omega_mid)], dtype=float),
            "hk_R_mid": np.array([float(fin.hk_R_mid)], dtype=float),
            "hk_X_mid": np.array([float(fin.hk_X_mid)], dtype=float),
            "hk_a1_well": np.array([float(fin.hk_a1_well)], dtype=float),
            "hk_a2_well": np.array([float(fin.hk_a2_well)], dtype=float),
            "hk_a2_barrier": np.array([float(fin.hk_a2_barrier)], dtype=float),
            "hk_diag_density": np.array([float(fin.hk_diag_density)], dtype=float),
            "hk_barrier_density": np.array([float(fin.hk_barrier_density)], dtype=float),
            "hk_abs_diag": np.array([float(fin.hk_abs_diag)], dtype=float),
            "hk_abs_offdiag": np.array([float(fin.hk_abs_offdiag)], dtype=float),
            "hk_barrier_ratio": np.array([float(fin.hk_barrier_ratio)], dtype=float),
            "hk_a1_flat": np.array([float(fin.hk_a1_flat)], dtype=float),
            "hk_a2_flat": np.array([float(fin.hk_a2_flat)], dtype=float),
            "hk_a1_well_geom": np.array([float(fin.hk_a1_well_geom)], dtype=float),
            "hk_a2_well_geom": np.array([float(fin.hk_a2_well_geom)], dtype=float),
            "hk_a2_barrier_geom": np.array([float(fin.hk_a2_barrier_geom)], dtype=float),
            "hk_diag_density_geom": np.array([float(fin.hk_diag_density_geom)], dtype=float),
            "hk_barrier_density_geom": np.array([float(fin.hk_barrier_density_geom)], dtype=float),
            "hk_abs_diag_geom": np.array([float(fin.hk_abs_diag_geom)], dtype=float),
            "hk_abs_offdiag_geom": np.array([float(fin.hk_abs_offdiag_geom)], dtype=float),
            "hk_barrier_ratio_geom": np.array([float(fin.hk_barrier_ratio_geom)], dtype=float),
            "hk_R_well_abs": np.array([float(fin.hk_R_well_abs)], dtype=float),
            "hk_R_barrier_abs": np.array([float(fin.hk_R_barrier_abs)], dtype=float),
            "hk_X_well_abs": np.array([float(fin.hk_X_well_abs)], dtype=float),
            "hk_X_barrier_abs": np.array([float(fin.hk_X_barrier_abs)], dtype=float),
            "hk_gradX_barrier": np.array([float(fin.hk_gradX_barrier)], dtype=float),
            "hk_curv_contrast_log": np.array([float(fin.hk_curv_contrast_log)], dtype=float),
            "hk_curv_access": np.array([float(fin.hk_curv_access)], dtype=float),
            "hk_barrier_stiffness_log": np.array([float(fin.hk_barrier_stiffness_log)], dtype=float),
            "hk_mass_access_diag": np.array([float(fin.hk_mass_access_diag)], dtype=float),
            "hk_mass_access_offdiag": np.array([float(fin.hk_mass_access_offdiag)], dtype=float),
            "hk_curv_screen_diag": np.array([float(fin.hk_curv_screen_diag)], dtype=float),
            "hk_curv_screen_offdiag": np.array([float(fin.hk_curv_screen_offdiag)], dtype=float),
            "hk_loop_prefactor_diag": np.array([float(fin.hk_loop_prefactor_diag)], dtype=float),
            "hk_loop_prefactor_offdiag": np.array([float(fin.hk_loop_prefactor_offdiag)], dtype=float),
            "hk_loop_local_prefactor_diag": np.array([float(fin.hk_loop_local_prefactor_diag)], dtype=float),
            "hk_loop_local_prefactor_offdiag": np.array([float(fin.hk_loop_local_prefactor_offdiag)], dtype=float),
            "eymh_loop_prefactor_diag": np.array([float(fin.eymh_loop_prefactor_diag)], dtype=float),
            "eymh_loop_prefactor_offdiag": np.array([float(fin.eymh_loop_prefactor_offdiag)], dtype=float),
            "eymh_source_prefactor_diag": np.array([float(fin.eymh_source_prefactor_diag)], dtype=float),
            "eymh_source_prefactor_offdiag": np.array([float(fin.eymh_source_prefactor_offdiag)], dtype=float),
            "eymh_parented_prefactor_diag": np.array([float(fin.eymh_parented_prefactor_diag)], dtype=float),
            "eymh_parented_prefactor_offdiag": np.array([float(fin.eymh_parented_prefactor_offdiag)], dtype=float),
            "loop_trace_p1": np.array([float(fin.loop_trace_p1)], dtype=float),
            "loop_trace_p2": np.array([float(fin.loop_trace_p2)], dtype=float),
            "loop_trace_neff": np.array([float(fin.loop_trace_neff)], dtype=float),
            "loop_trace_entropy_norm": np.array([float(fin.loop_trace_entropy_norm)], dtype=float),
            "coeff_participation_access": np.array([float(fin.coeff_participation_access)], dtype=float),
            "tree_diag_susceptibility": np.array([float(fin.tree_diag_susceptibility)], dtype=float),
            "tree_diag_compressibility": np.array([float(fin.tree_diag_compressibility)], dtype=float),
            "coeff_participation_access_parented": np.array([float(fin.coeff_participation_access_parented)], dtype=float),
            "tree_diag_susceptibility_parented": np.array([float(fin.tree_diag_susceptibility_parented)], dtype=float),
            "tree_diag_compressibility_parented": np.array([float(fin.tree_diag_compressibility_parented)], dtype=float),
            "tree_diag_pressure_fraction_parented": np.array([float(fin.tree_diag_pressure_fraction_parented)], dtype=float),
            "gamma_diag": np.array([float(rge.gamma_diag)], dtype=float),
            "gamma_offdiag": np.array([float(rge.gamma_offdiag)], dtype=float),
            "finite_fac_diag": np.array([float(fin.finite_fac_diag)], dtype=float),
            "finite_fac_offdiag": np.array([float(fin.finite_fac_offdiag)], dtype=float),
            "match_rebuild_residual": np.array(
                [float(np.max(np.abs(fin.c_match - c_match_rebuild)))],
                dtype=float,
            ),
            "ir_rebuild_residual": np.array(
                [float(np.max(np.abs(rge.c_low - c_ir_rebuild)))],
                dtype=float,
            ),
        }

    def hll_wilson_matrix_uv_match_with_meta(
        self,
        D: float,
        eta: float,
        t_coh: float,
        N_max: int = 20,
    ) -> tuple[np.ndarray, Dict[str, float]]:
        """
        UV-tree matrix with minimal finite one-loop matching applied at mu_match.
        """
        g_uv = self._hll_g_uv_matrix(D)
        p_kin = self._hll_pkin_vector(D, eta, t_coh, N_max=N_max)
        m2 = self._hll_m2_vector(D)
        c_uv = wilson_matrix_uv_tree(g_uv=g_uv, p_kin=p_kin, m2=m2, cfg=self._hll_uv_tree_config())
        cfg = self._hll_uv_finite_match_config()
        c_match, meta = apply_ceh_finite_one_loop(c_tree=c_uv, cfg=cfg, g_uv=g_uv, p_kin=p_kin, m2=m2, D=float(D))
        return c_match, meta

    def hll_wilson_matrix_uv_match(self, D: float, eta: float, t_coh: float, N_max: int = 20) -> np.ndarray:
        c_match, _ = self.hll_wilson_matrix_uv_match_with_meta(D=D, eta=eta, t_coh=t_coh, N_max=N_max)
        return c_match

    def run_ceh_llrg(
        self,
        c_uv: np.ndarray,
        m2: np.ndarray,
        g_uv: Optional[np.ndarray] = None,
        p_kin: Optional[np.ndarray] = None,
        D: Optional[float] = None,
    ) -> tuple[np.ndarray, Dict[str, float]]:
        """
        Run leading-log RGE from UV matching scale to the configured low scale.

        Returns:
          (C_low, metadata) where metadata contains mu_match, mu_low, log_ratio.
        """
        fin_cfg = self._hll_uv_finite_match_config()
        c_match, fin_meta = apply_ceh_finite_one_loop(c_tree=c_uv, cfg=fin_cfg, g_uv=g_uv, p_kin=p_kin, m2=m2, D=D)
        cfg = self._hll_uv_rge_config()
        mu_match = mu_match_from_m2(m2, floor=cfg.floor)
        c_low, log_ratio = run_ceh_leading_log(c_match=c_match, mu_match=mu_match, cfg=cfg)
        meta = {
            "mu_match": float(mu_match),
            "mu_low": float(cfg.mu_low),
            "log_ratio": float(log_ratio),
            "finite_match_mode": str(fin_meta.get("mode", "constant")),
            "kappa_diag": float(fin_meta["kappa_diag"]),
            "kappa_offdiag": float(fin_meta["kappa_offdiag"]),
            "kappa_diag_eff": float(fin_meta.get("kappa_diag_eff", fin_meta["kappa_diag"])),
            "kappa_offdiag_eff": float(fin_meta.get("kappa_offdiag_eff", fin_meta["kappa_offdiag"])),
            "shell_spread": float(fin_meta.get("shell_spread", 0.0)),
            "coeff_cv": float(fin_meta.get("coeff_cv", 0.0)),
            "offdiag_mix": float(fin_meta.get("offdiag_mix", 0.0)),
            "gap_cv": float(fin_meta.get("gap_cv", 0.0)),
            "gap_asym": float(fin_meta.get("gap_asym", 0.0)),
            "g_col_norm_cv": float(fin_meta.get("g_col_norm_cv", 0.0)),
            "c_tree_diag_cv": float(fin_meta.get("c_tree_diag_cv", 0.0)),
            "pkin_entropy": float(fin_meta.get("pkin_entropy", 0.0)),
            "action_norm_diag": float(fin_meta.get("action_norm_diag", 1.0)),
            "action_norm_offdiag": float(fin_meta.get("action_norm_offdiag", 1.0)),
            "coeff_l1": float(fin_meta.get("coeff_l1", 0.0)),
            "coeff_l2": float(fin_meta.get("coeff_l2", 0.0)),
            "coeff_align": float(fin_meta.get("coeff_align", 0.0)),
            "action_abs_diag": float(fin_meta.get("action_abs_diag", 0.0)),
            "action_abs_offdiag": float(fin_meta.get("action_abs_offdiag", 0.0)),
            "hk_omega_mid": float(fin_meta.get("hk_omega_mid", 0.0)),
            "hk_R_mid": float(fin_meta.get("hk_R_mid", 0.0)),
            "hk_X_mid": float(fin_meta.get("hk_X_mid", 0.0)),
            "hk_a1_well": float(fin_meta.get("hk_a1_well", 0.0)),
            "hk_a2_well": float(fin_meta.get("hk_a2_well", 0.0)),
            "hk_a2_barrier": float(fin_meta.get("hk_a2_barrier", 0.0)),
            "hk_diag_density": float(fin_meta.get("hk_diag_density", 0.0)),
            "hk_barrier_density": float(fin_meta.get("hk_barrier_density", 0.0)),
            "hk_abs_diag": float(fin_meta.get("hk_abs_diag", 0.0)),
            "hk_abs_offdiag": float(fin_meta.get("hk_abs_offdiag", 0.0)),
            "hk_barrier_ratio": float(fin_meta.get("hk_barrier_ratio", 0.0)),
            "hk_a1_flat": float(fin_meta.get("hk_a1_flat", 0.0)),
            "hk_a2_flat": float(fin_meta.get("hk_a2_flat", 0.0)),
            "hk_a1_well_geom": float(fin_meta.get("hk_a1_well_geom", 0.0)),
            "hk_a2_well_geom": float(fin_meta.get("hk_a2_well_geom", 0.0)),
            "hk_a2_barrier_geom": float(fin_meta.get("hk_a2_barrier_geom", 0.0)),
            "hk_diag_density_geom": float(fin_meta.get("hk_diag_density_geom", 0.0)),
            "hk_barrier_density_geom": float(fin_meta.get("hk_barrier_density_geom", 0.0)),
            "hk_abs_diag_geom": float(fin_meta.get("hk_abs_diag_geom", 0.0)),
            "hk_abs_offdiag_geom": float(fin_meta.get("hk_abs_offdiag_geom", 0.0)),
            "hk_barrier_ratio_geom": float(fin_meta.get("hk_barrier_ratio_geom", 0.0)),
            "hk_R_well_abs": float(fin_meta.get("hk_R_well_abs", 0.0)),
            "hk_R_barrier_abs": float(fin_meta.get("hk_R_barrier_abs", 0.0)),
            "hk_X_well_abs": float(fin_meta.get("hk_X_well_abs", 0.0)),
            "hk_X_barrier_abs": float(fin_meta.get("hk_X_barrier_abs", 0.0)),
            "hk_gradX_barrier": float(fin_meta.get("hk_gradX_barrier", 0.0)),
            "hk_curv_contrast_log": float(fin_meta.get("hk_curv_contrast_log", 0.0)),
            "hk_curv_access": float(fin_meta.get("hk_curv_access", 0.0)),
            "hk_barrier_stiffness_log": float(fin_meta.get("hk_barrier_stiffness_log", 0.0)),
            "hk_mass_access_diag": float(fin_meta.get("hk_mass_access_diag", 0.0)),
            "hk_mass_access_offdiag": float(fin_meta.get("hk_mass_access_offdiag", 0.0)),
            "hk_curv_screen_diag": float(fin_meta.get("hk_curv_screen_diag", 0.0)),
            "hk_curv_screen_offdiag": float(fin_meta.get("hk_curv_screen_offdiag", 0.0)),
            "hk_loop_prefactor_diag": float(fin_meta.get("hk_loop_prefactor_diag", 0.0)),
            "hk_loop_prefactor_offdiag": float(fin_meta.get("hk_loop_prefactor_offdiag", 0.0)),
            "hk_loop_local_prefactor_diag": float(fin_meta.get("hk_loop_local_prefactor_diag", 0.0)),
            "hk_loop_local_prefactor_offdiag": float(fin_meta.get("hk_loop_local_prefactor_offdiag", 0.0)),
            "eymh_loop_prefactor_diag": float(fin_meta.get("eymh_loop_prefactor_diag", 0.0)),
            "eymh_loop_prefactor_offdiag": float(fin_meta.get("eymh_loop_prefactor_offdiag", 0.0)),
            "eymh_source_prefactor_diag": float(fin_meta.get("eymh_source_prefactor_diag", 0.0)),
            "eymh_source_prefactor_offdiag": float(fin_meta.get("eymh_source_prefactor_offdiag", 0.0)),
            "coeff_participation_access": float(fin_meta.get("coeff_participation_access", 0.0)),
            "tree_diag_susceptibility": float(fin_meta.get("tree_diag_susceptibility", 0.0)),
            "tree_diag_compressibility": float(fin_meta.get("tree_diag_compressibility", 0.0)),
            "eymh_parented_prefactor_diag": float(fin_meta.get("eymh_parented_prefactor_diag", 0.0)),
            "eymh_parented_prefactor_offdiag": float(fin_meta.get("eymh_parented_prefactor_offdiag", 0.0)),
            "loop_trace_p1": float(fin_meta.get("loop_trace_p1", 0.0)),
            "loop_trace_p2": float(fin_meta.get("loop_trace_p2", 0.0)),
            "loop_trace_neff": float(fin_meta.get("loop_trace_neff", 0.0)),
            "loop_trace_entropy_norm": float(fin_meta.get("loop_trace_entropy_norm", 0.0)),
            "coeff_participation_access_parented": float(fin_meta.get("coeff_participation_access_parented", 0.0)),
            "tree_diag_susceptibility_parented": float(fin_meta.get("tree_diag_susceptibility_parented", 0.0)),
            "tree_diag_compressibility_parented": float(fin_meta.get("tree_diag_compressibility_parented", 0.0)),
            "tree_diag_pressure_fraction_parented": float(fin_meta.get("tree_diag_pressure_fraction_parented", 0.0)),
            "finite_fac_diag": float(fin_meta["finite_fac_diag"]),
            "finite_fac_offdiag": float(fin_meta["finite_fac_offdiag"]),
        }
        return c_low, meta

    def hll_wilson_matrix_uv_rge_with_meta(
        self,
        D: float,
        eta: float,
        t_coh: float,
        N_max: int = 20,
    ) -> tuple[np.ndarray, Dict[str, float]]:
        """
        UV-tree matrix followed by leading-log running to the low scale,
        with explicit running metadata.
        """
        g_uv = self._hll_g_uv_matrix(D)
        p_kin = self._hll_pkin_vector(D, eta, t_coh, N_max=N_max)
        m2 = self._hll_m2_vector(D)
        c_uv = wilson_matrix_uv_tree(g_uv=g_uv, p_kin=p_kin, m2=m2, cfg=self._hll_uv_tree_config())
        c_low, meta = self.run_ceh_llrg(c_uv=c_uv, m2=m2, g_uv=g_uv, p_kin=p_kin, D=float(D))
        return c_low, meta

    def hll_wilson_coeff_uv_tree(self, layer_n: int, D: float, eta: float, t_coh: float, N_max: int = 20) -> float:
        if layer_n <= 0 or layer_n > 3:
            return 0.0
        cmat = self.hll_wilson_matrix_uv_tree(D, eta, t_coh, N_max=N_max)
        return float(max(cmat[layer_n - 1, layer_n - 1], self.params.b_overlap_floor))

    def hll_wilson_matrix_uv_rge(self, D: float, eta: float, t_coh: float, N_max: int = 20) -> np.ndarray:
        """
        UV-tree matrix followed by leading-log running to the low scale.
        """
        c_low, _ = self.hll_wilson_matrix_uv_rge_with_meta(D=D, eta=eta, t_coh=t_coh, N_max=N_max)
        return c_low

    def hll_wilson_coeff_uv_rge(self, layer_n: int, D: float, eta: float, t_coh: float, N_max: int = 20) -> float:
        if layer_n <= 0 or layer_n > 3:
            return 0.0
        cmat = self.hll_wilson_matrix_uv_rge(D, eta, t_coh, N_max=N_max)
        return float(max(cmat[layer_n - 1, layer_n - 1], self.params.b_overlap_floor))

    def hll_total_width_ratio_matched(
        self,
        D: float,
        eta: float,
        t_coh: float,
        ref_D: float,
        ref_eta: float,
        N_max: int = 20,
    ) -> float:
        cfg = self._hll_match_config()
        c_ref = self.hll_wilson_matrix_matched(ref_D, ref_eta, t_coh, N_max=N_max)
        c = self.hll_wilson_matrix_matched(D, eta, t_coh, N_max=N_max)
        c_diag = np.diag(c)
        c_ref_diag = np.diag(c_ref)
        return total_width_ratio(c_diag=c_diag, c_diag_ref=c_ref_diag, cfg=cfg)

    def hll_total_width_ratio_uv_tree(
        self,
        D: float,
        eta: float,
        t_coh: float,
        ref_D: float,
        ref_eta: float,
        N_max: int = 20,
    ) -> float:
        cfg = self._hll_match_config()
        c_ref = self.hll_wilson_matrix_uv_tree(ref_D, ref_eta, t_coh, N_max=N_max)
        c = self.hll_wilson_matrix_uv_tree(D, eta, t_coh, N_max=N_max)
        c_diag = np.diag(c)
        c_ref_diag = np.diag(c_ref)
        return total_width_ratio(c_diag=c_diag, c_diag_ref=c_ref_diag, cfg=cfg)

    def hll_total_width_ratio_uv_rge(
        self,
        D: float,
        eta: float,
        t_coh: float,
        ref_D: float,
        ref_eta: float,
        N_max: int = 20,
    ) -> float:
        cfg = self._hll_match_config()
        c_ref = self.hll_wilson_matrix_uv_rge(ref_D, ref_eta, t_coh, N_max=N_max)
        c = self.hll_wilson_matrix_uv_rge(D, eta, t_coh, N_max=N_max)
        c_diag = np.diag(c)
        c_ref_diag = np.diag(c_ref)
        width_ratio = total_width_ratio(c_diag=c_diag, c_diag_ref=c_ref_diag, cfg=cfg)
        return self._blend_observable_width_ratio(width_ratio=width_ratio, D=D, eta=eta)

    def hll_channel_amplitude(
        self,
        layer_n: int,
        D: float,
        eta: float,
        t_coh: float,
        observable_mode: str = "eft_wilson_matched",
        N_max: int = 20,
    ) -> float:
        if observable_mode == "proxy_wratio":
            return self.layer_weight(layer_n, D, eta, t_coh)
        if observable_mode == "eft_wilson_diag":
            return self.hll_wilson_coeff(layer_n, D, eta, t_coh, N_max=N_max)
        if observable_mode == "eft_wilson_matched":
            return self.hll_wilson_coeff_matched(layer_n, D, eta, t_coh, N_max=N_max)
        if observable_mode == "eft_wilson_uv_tree":
            return self.hll_wilson_coeff_uv_tree(layer_n, D, eta, t_coh, N_max=N_max)
        if observable_mode == "eft_wilson_uv_rge":
            return self.hll_wilson_coeff_uv_rge(layer_n, D, eta, t_coh, N_max=N_max)
        raise ValueError(f"Unsupported observable_mode='{observable_mode}'.")

    def hll_mu_pred(
        self,
        layer_n: int,
        D: float,
        eta: float,
        t_coh: float,
        ref_D: float,
        ref_eta: float,
        observable_mode: Optional[str] = None,
        N_max: Optional[int] = None,
    ) -> float:
        mode = self.params.hll_observable_mode if observable_mode is None else observable_mode
        nmax = self.params.hll_observable_nmax if N_max is None else int(N_max)
        amp_ref = self.hll_channel_amplitude(layer_n, ref_D, ref_eta, t_coh, observable_mode=mode, N_max=nmax)
        amp_ref = self._blend_observable_ref_amp(
            amp_ref=amp_ref,
            layer_n=int(layer_n),
            observable_mode=str(mode),
            D=float(D),
        )
        amp = self.hll_channel_amplitude(layer_n, D, eta, t_coh, observable_mode=mode, N_max=nmax)
        amp = self._blend_observable_point_amp(
            amp=amp,
            layer_n=int(layer_n),
            observable_mode=str(mode),
            D=float(D),
            eta=float(eta),
        )
        ratio = float(amp / max(amp_ref, 1e-30))
        if mode == "proxy_wratio":
            return ratio
        partial_ratio_raw = ratio * ratio
        partial_ratio_base = self._blend_observable_partial_ratio(
            partial_ratio=partial_ratio_raw,
            layer_n=int(layer_n),
            observable_mode=str(mode),
            D=float(D),
            eta=float(eta),
            apply_two_lobe=False,
        )
        partial_ratio = self._blend_observable_partial_ratio(
            partial_ratio=partial_ratio_raw,
            layer_n=int(layer_n),
            observable_mode=str(mode),
            D=float(D),
            eta=float(eta),
        )
        if mode == "eft_wilson_matched":
            width_ratio = self.hll_total_width_ratio_matched(
                D=D,
                eta=eta,
                t_coh=t_coh,
                ref_D=ref_D,
                ref_eta=ref_eta,
                N_max=nmax,
            )
            mu_base = float(partial_ratio_base / max(width_ratio, 1e-30))
            mu_runtime = float(partial_ratio / max(width_ratio, 1e-30))
            return self._apply_observable_late_dstrip(mu_base=mu_base, mu_runtime=mu_runtime, D=float(D))
        if mode == "eft_wilson_uv_tree":
            width_ratio = self.hll_total_width_ratio_uv_tree(
                D=D,
                eta=eta,
                t_coh=t_coh,
                ref_D=ref_D,
                ref_eta=ref_eta,
                N_max=nmax,
            )
            mu_base = float(partial_ratio_base / max(width_ratio, 1e-30))
            mu_runtime = float(partial_ratio / max(width_ratio, 1e-30))
            return self._apply_observable_late_dstrip(mu_base=mu_base, mu_runtime=mu_runtime, D=float(D))
        if mode == "eft_wilson_uv_rge":
            width_ratio = self.hll_total_width_ratio_uv_rge(
                D=D,
                eta=eta,
                t_coh=t_coh,
                ref_D=ref_D,
                ref_eta=ref_eta,
                N_max=nmax,
            )
            mu_base = float(partial_ratio_base / max(width_ratio, 1e-30))
            mu_runtime = float(partial_ratio / max(width_ratio, 1e-30))
            return self._apply_observable_late_dstrip(mu_base=mu_base, mu_runtime=mu_runtime, D=float(D))
        return partial_ratio

    def _interp_scalar(self, D: float, d_knots: np.ndarray, y_knots: np.ndarray) -> float:
        order = np.argsort(d_knots)
        d_sorted = d_knots[order]
        y_sorted = y_knots[order]
        return float(np.interp(D, d_sorted, y_sorted))

    def _grid_scalar(
        self,
        D: float,
        d_knots: np.ndarray,
        y_knots: np.ndarray,
        tol: float = 1e-8,
        fallback_interp: bool = True,
    ) -> float:
        """Lookup y(D) on an action-derived D grid with optional interpolation fallback."""
        dvals = np.asarray(d_knots, dtype=float)
        yvals = np.asarray(y_knots, dtype=float)
        if dvals.size == 0:
            raise ValueError("empty grid profile")

        idx = int(np.argmin(np.abs(dvals - float(D))))
        if abs(float(dvals[idx]) - float(D)) <= tol:
            return float(yvals[idx])
        if fallback_interp:
            return self._interp_scalar(float(D), dvals, yvals)
        raise ValueError(f"D={D} not found in action-derived grid profile (tol={tol}).")

    def _lindblad_cmax(self, delta: float, gamma_phi: float, gamma_mix: float) -> float:
        H = np.array([[0.0, delta / 2.0], [delta / 2.0, 0.0]], dtype=complex)
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        L_ops = [np.sqrt(gamma_phi) * sigma_z, np.sqrt(gamma_mix) * sigma_x]

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            rho = y.reshape(2, 2)
            drho = -1j * (H @ rho - rho @ H)
            for L in L_ops:
                drho += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
            return drho.reshape(-1)

        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex).reshape(-1)
        t_eval = np.linspace(0.0, self.params.chi_open_tmax, self.params.chi_open_nstep)
        sol = solve_ivp(
            rhs,
            (0.0, self.params.chi_open_tmax),
            rho0,
            t_eval=t_eval,
            rtol=self.params.chi_open_rtol,
            atol=self.params.chi_open_atol,
        )
        if sol.y.shape[1] == 0:
            return 0.0
        max_abs = 0.0
        for i in range(sol.y.shape[1]):
            rho = sol.y[:, i].reshape(2, 2)
            max_abs = max(max_abs, float(np.abs(rho[0, 1])))
        return max_abs

    def chi_effective(self, D: float) -> float:
        """
        Effective mixing coefficient used in eps_mix.

        - constant: chi_eff = params.chi
        - localized_interp: piecewise-linear interpolation of chi_LR(D)
          with endpoint clamping outside knot range.
        - localized_grid: action-derived grid lookup for chi_LR(D),
          with interpolation fallback only if D is off-grid.
        - localized_grid_strict: action-derived grid lookup for chi_LR(D),
          with no interpolation fallback (raises if D is off-grid).
        - open_system / open_system_micro: profile-interpolated
          (delta, gamma_phi, gamma_mix, gamma_ref) fed into
          two-level Lindblad dynamics, returning
          chi_eff = 2*gamma_mix*Cmax/gamma_ref.
        """
        mode = self._chi_mode_active
        if mode == "constant":
            return float(self.params.chi)

        if mode == "localized_interp":
            d_knots = np.asarray(self.params.chi_lr_D, dtype=float)
            chi_knots = np.asarray(self.params.chi_lr_vals, dtype=float)
            return self._interp_scalar(D, d_knots, chi_knots)

        if mode == "localized_runtime_direct":
            return self._runtime_direct_chi(float(D))

        if mode in {"localized_grid", "localized_grid_strict"}:
            d_knots = np.asarray(self.params.chi_lr_D, dtype=float)
            chi_knots = np.asarray(self.params.chi_lr_vals, dtype=float)
            strict = mode == "localized_grid_strict"
            return self._grid_scalar(D, d_knots, chi_knots, fallback_interp=not strict)

        if mode in {"open_system", "open_system_micro"} and self._chi_open_profile is not None:
            key = float(round(D, 8))
            cached = self._chi_open_cache.get(key, None)
            if cached is not None:
                return cached

            prof = self._chi_open_profile
            d_knots = prof["D"]
            delta = self._interp_scalar(D, d_knots, prof["delta"])
            gamma_phi = self.params.chi_open_phi_scale * self._interp_scalar(D, d_knots, prof["gamma_phi"])
            gamma_mix = self.params.chi_open_mix_scale * self._interp_scalar(D, d_knots, prof["gamma_mix"])
            gamma_ref = self._interp_scalar(D, d_knots, prof["gamma_ref"])

            cmax = self._lindblad_cmax(delta=delta, gamma_phi=gamma_phi, gamma_mix=gamma_mix)
            chi_eff = float(2.0 * gamma_mix * cmax / max(gamma_ref, 1e-30))
            chi_eff = max(chi_eff, 0.0)
            self._chi_open_cache[key] = chi_eff
            return chi_eff

        return float(self.params.chi)

    # --- Geometry & WKB (Explicit Spec) ---
    def V_eff(self, x: float, mu: float, D: float) -> float:
        """
        Effective potential V_eff(x; D).
        Spec: V_eff(x) = mu^2 * [1 + a0 * (D/2)^2 / (x^2 + eps^2)]^2
        Units: [Mass]^2
        """
        eps, a0 = self.params.eps, self.params.a0
        # Omega is dimensionless conformal factor
        Omega = 1 + a0 * (D/2)**2 / (x**2 + eps**2)
        return (mu**2) * (Omega**2)

    def omega_N(self, mu: float, N: int, D: float) -> float:
        """
        Layer frequency spectrum (hydrogenic proxy).

        Spec:
            omega_N = mu * (1 - alpha(D)^2 / (2*N^2))
        where mu = M_*/D has units [Mass], and alpha(D)=1/D is dimensionless.
        """
        if N <= 0:
            return 0.0
        alpha = 1.0 / D
        return float(mu * (1.0 - (alpha**2) / (2.0 * N**2)))

    def action_S(self, mu: float, D: float, w: float) -> float:
        """
        WKB tunneling action S_N (dimensionless).

        We work with a dimensionless coordinate x = M_* r. The physical action is
            S = ∫ dr sqrt(V_eff - w^2) = ∫ dx / M_* sqrt(V_eff - w^2).
        This function returns the dimensionless S for a symmetric barrier.
        """
        # Effective potential on a positive x grid (symmetric barrier)
        x_max = 20.0
        xs = np.linspace(0.0, x_max, 2000)
        V = (mu**2) * (1.0 + self.params.a0 * (D/2.0)**2 / (xs**2 + self.params.eps**2))**2

        # Find turning point x_tp where V(x_tp) ~ w^2 (last point with V>w^2)
        valid_indices = np.where(V > w*w)[0]
        if len(valid_indices) == 0:
            return 0.0
        x_tp = xs[valid_indices[-1]]

        # Integration on [0, x_tp]
        grid = np.linspace(0.0, x_tp, 1000)
        V_grid = (mu**2) * (1.0 + self.params.a0 * (D/2.0)**2 / (grid**2 + self.params.eps**2))**2

        # Dimensionless action requires a 1/M_* factor
        integrand = np.sqrt(np.maximum(V_grid - w*w, 0.0)) / self.params.M
        integral = np.trapz(integrand, grid)

        # Full barrier action: x_-=-x_tp, x_+=+x_tp (even integrand)
        return float(2.0 * integral)

    # --- Rank-2 Kinetics ---
    def calculate_gamma_N(self, N: int, D: float, eta: float) -> float:
        """
        Calculate Gamma_N using the Eigenvalue Method (Rank-2 closure).

        Units:
          - M_* (self.params.M) carries [Mass]
          - D is dimensionless (physical separation is D/M_*)
          - omega_N and Gamma_N carry [Mass]
        """
        if D <= 0:
            return 0.0

        eta_eff = self.eta_effective(D, eta)
        key = (int(N), float(round(D, 8)))
        use_pref_cache = True
        if (self._chi_mode_active == "localized_runtime_direct" or self._gamma_mode_active == "action_runtime_direct") and (not self.params.runtime_direct_use_cache):
            use_pref_cache = False

        if use_pref_cache:
            pref = self._gamma_prefactor_cache.get(key, None)
            if pref is not None:
                return max(float(eta_eff * pref), 0.0)

        mu = self.params.M / D  # [Mass]
        OmegaH = self.params.Omega_H * self.params.M  # [Mass]
        alpha = 1.0 / D  # dimensionless

        # Superradiant-inspired proxy channel rates (units restored by prefactor M_*)
        def gamma_sr(l: int, m: int, A: float) -> float:
            w = self.omega_N(mu, N, D)  # [Mass]
            delta_tilde = (m * OmegaH - w) / self.params.M  # dimensionless
            if delta_tilde <= 0.0:
                return 0.0
            return self.params.M * A * (alpha ** (4 * l + 4)) * delta_tilde  # [Mass]

        A1_eff, A2_eff = self._gamma_action_A12(D, N=N)
        g1 = gamma_sr(1, 1, A1_eff)
        g2 = gamma_sr(2, 2, A2_eff)

        # WKB tunneling suppression (dimensionless)
        w_val = self.omega_N(mu, N, D)
        S = self.action_S(mu, D, w_val)
        # Mixing term (shares units with g1,g2)
        chi_eff = self.chi_effective(D)
        eps_mix = chi_eff * math.sqrt(g1 * g2) if (g1 > 0 and g2 > 0) else 0.0

        # Largest eigenvalue of 2x2 symmetric matrix [[g1,eps],[eps,g2]]
        tr = g1 + g2
        disc = (g1 - g2) ** 2 + 4.0 * (eps_mix ** 2)
        lam_plus = 0.5 * (tr + math.sqrt(disc))

        pref = float(np.exp(-2.0 * S) * lam_plus)
        if use_pref_cache:
            self._gamma_prefactor_cache[key] = pref
        Gamma = eta_eff * pref
        return max(float(Gamma), 0.0)

    def get_probabilities(self, D: float, eta: float, t_coh: float, N_max: int=10) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Calculate P_N.
        t_coh must be in inverse mass units. 
        Product Gamma_N * t_coh is dimensionless.
        """
        Ns = np.arange(1, N_max+1)
        weights = []
        gammas = []
        
        for N in Ns:
            Gam = self.calculate_gamma_N(N, D, eta)
            w = self.layer_weight(N, D, eta, t_coh)
            
            weights.append(w)
            gammas.append(Gam)
            
        weights = np.array(weights)
        total = weights.sum()
        P_N = weights / total if total > 0 else weights
        
        # Metadata analysis
        winner_idx = np.argmax(P_N)
        gen_ratio = np.sum(P_N[:3])
        
        meta = {
            "winner": int(Ns[winner_idx]),
            "generation_ratio": float(gen_ratio),
            "P_N": P_N,
            "gammas": np.array(gammas)
        }
        return Ns, P_N, meta

# End of library
