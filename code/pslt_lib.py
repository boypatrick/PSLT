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
    mixing_epsilon,
    wilson_matrix,
    wilson_matrix_uv_tree,
    total_width_ratio,
)
from eft_rge import (
    EFTFiniteOneLoopMatchConfig,
    EFTLeadingLogRGEConfig,
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
    g_mode: str = "cardy"   # "cardy", "fp_1d", "fp_2d", "fp_1d_full", "fp_2d_full"
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
    
    # Geometry & Kinetics
    Omega_H: float = 0.9    # Horizon proxy angular velocity [Mass] (scaled by M)
    A1: float = 1.0         # l=1 amplitude (dimensionless prefactor for rate)
    A2: float = 1.0         # l=2 amplitude (dimensionless prefactor for rate)
    gamma_mode: str = "surrogate"  # "surrogate", "action_profile", "action_grid", "action_grid_strict", or "action_runtime_direct"
    gamma_superrad_csv: Optional[str] = None
    gamma_superrad_scale: float = 1.0
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
    b_mode: str = "yukawa"  # "yukawa" or "overlap_2d"
    b_overlap_csv: Optional[str] = None
    b_overlap_floor: float = 1e-8
    b_n_mode: str = "cumulative"  # "cumulative" or "single" over lepton Yukawas
    b_n_power: float = 0.30       # Sublinear compression: B_gen ∝ (y_gen)^{b_n_power}
    b_n_tail_mode: str = "saturate"  # "saturate" (paper baseline) or "gaussian"
    b_n_tail_beta: float = 0.50   # Used only when b_n_tail_mode == "gaussian"
    hll_observable_mode: str = "eft_wilson_uv_rge"  # "proxy_wratio", "eft_wilson_diag", "eft_wilson_matched", "eft_wilson_uv_tree", or "eft_wilson_uv_rge"
    hll_observable_nmax: int = 20
    hll_match_basis_mode: str = "sqrt_yraw"  # "sqrt_yraw" reproduces diagonal limit with mix_scale=0
    hll_match_mix_scale: float = 200.0
    hll_match_mix_max: float = 0.25
    hll_match_eta_power: float = 1.0
    hll_match_eta_ref: float = 1.0
    hll_match_width_mode: str = "sm_leptonic"  # "none" or "sm_leptonic"
    hll_match_width_scale: float = 1.0
    hll_match_br_ee: float = 5.0e-9
    hll_match_br_mumu: float = 2.2e-4
    hll_match_br_tautau: float = 6.3e-2
    hll_uv_m2_floor: float = 1e-10
    hll_uv_coupling_floor: float = 1e-30
    hll_uv_blend: float = 0.0
    hll_uv_m2_power: float = 1.0
    hll_uv_match_kappa_diag: float = 0.0
    hll_uv_match_kappa_offdiag: float = 0.0
    hll_uv_rge_mu_low: float = 1.0
    hll_uv_rge_gamma_diag: float = 2.0
    hll_uv_rge_gamma_offdiag: float = 1.0
    hll_uv_rge_log_clip: float = 6.0

    def __post_init__(self):
        if self.chi_mode not in {"constant", "localized_interp", "localized_grid", "localized_grid_strict", "localized_runtime_direct", "open_system", "open_system_micro"}:
            raise ValueError(f"Unsupported chi_mode='{self.chi_mode}'.")
        if self.gamma_mode not in {"surrogate", "action_profile", "action_grid", "action_grid_strict", "action_runtime_direct"}:
            raise ValueError(f"Unsupported gamma_mode='{self.gamma_mode}'.")
        if self.gamma_superrad_scale <= 0:
            raise ValueError("gamma_superrad_scale must be > 0.")
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
        if self.g_mode not in {"cardy", "fp_1d", "fp_2d", "fp_1d_full", "fp_2d_full"}:
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
        if self.b_mode not in {"yukawa", "overlap_2d"}:
            raise ValueError(f"Unsupported b_mode='{self.b_mode}'.")
        if self.b_overlap_floor <= 0:
            raise ValueError("b_overlap_floor must be > 0.")
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
        if self.hll_match_basis_mode not in {"sqrt_yraw", "yraw"}:
            raise ValueError(f"Unsupported hll_match_basis_mode='{self.hll_match_basis_mode}'.")
        if self.hll_match_mix_scale < 0.0:
            raise ValueError("hll_match_mix_scale must be >= 0.")
        if not (0.0 <= self.hll_match_mix_max <= 0.49):
            raise ValueError("hll_match_mix_max must be in [0, 0.49].")
        if self.hll_match_eta_ref <= 0.0:
            raise ValueError("hll_match_eta_ref must be > 0.")
        if self.hll_match_width_mode not in {"none", "sm_leptonic"}:
            raise ValueError(f"Unsupported hll_match_width_mode='{self.hll_match_width_mode}'.")
        if self.hll_match_width_scale < 0.0:
            raise ValueError("hll_match_width_scale must be >= 0.")
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
        if self.hll_uv_rge_mu_low <= 0.0:
            raise ValueError("hll_uv_rge_mu_low must be > 0.")
        if self.hll_uv_rge_log_clip <= 0.0:
            raise ValueError("hll_uv_rge_log_clip must be > 0.")

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
        self._gamma_eta_profile: Optional[Dict[str, np.ndarray]] = None
        self._gamma_eta_mode_active: str = "scan"
        self._tcoh_mode_active: str = "input"
        self._tcoh_profile: Optional[Dict[str, np.ndarray]] = None
        self._b_mode_active: str = "yukawa"
        self._b_overlap_profile: Optional[Dict[str, np.ndarray]] = None
        
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
        d_key = float(round(D, 8))
        cached = self._g_fp_2d_full_hat_cache.get(d_key)
        if cached is not None:
            return cached

        spec = self._interp_g_fp_2d_spectrum(D)
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
            self._g_fp_2d_full_hat_cache[d_key] = hat
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

        self._g_fp_2d_full_hat_cache[d_key] = hat
        return hat

    def _init_g_profiles(self) -> None:
        p1 = Path(self.params.g_fp_1d_csv) if self.params.g_fp_1d_csv else self.root_dir / "output" / "gn_fp_1d" / "gn_phase_space_candidate_D12.csv"
        p2 = Path(self.params.g_fp_2d_csv) if self.params.g_fp_2d_csv else self.root_dir / "output" / "gn_fp_2d" / "gn_phase_space_2d_D6-12-18.csv"
        p2_spec = Path(self.params.g_fp_2d_spectrum_csv) if self.params.g_fp_2d_spectrum_csv else self.root_dir / "output" / "gn_fp_2d" / "gn_phase_space_2d_spectrum_D6-12-18.csv"

        self._g_fp_1d_profile = self._load_g_fp_1d_profile(p1)
        self._g_fp_2d_profile = self._load_g_fp_2d_profile(p2)
        self._g_fp_2d_spectrum = self._load_g_fp_2d_spectrum(p2_spec)
        self._g_fp_2d_spectrum_interp_cache.clear()
        self._g_fp_2d_full_hat_cache.clear()

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

        if mode != "overlap_2d":
            return

        if self.params.b_overlap_csv:
            path = Path(self.params.b_overlap_csv)
        else:
            path = self._auto_find_b_overlap_csv()

        if path is None:
            print("Warning: b_mode=overlap_2d requested but no overlap profile CSV was found. Falling back to yukawa.")
            return

        prof = self._load_b_overlap_profile(path)
        if prof is None:
            print(f"Warning: could not parse overlap B_N profile from {path}. Falling back to yukawa.")
            return

        self._b_overlap_profile = prof
        self._b_mode_active = "overlap_2d"

    def active_b_mode(self) -> str:
        return self._b_mode_active

    def active_g_mode(self) -> str:
        return self._g_mode_active

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

    def _gamma_action_A12(self, D: float) -> Tuple[float, float]:
        if self._gamma_mode_active == "action_runtime_direct":
            a1, a2 = self._runtime_direct_a12(float(D))
            scale = float(self.params.gamma_superrad_scale)
            return max(a1 * scale, 1e-30), max(a2 * scale, 1e-30)

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
        """
        if N <= 0:
            return 0.0

        mode = self._g_mode_active
        if mode == "cardy":
            return self.g_N_cardy(N)

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

    def y_eff_raw_N(self, N: int, D: Optional[float] = None) -> float:
        """
        Effective overlap amplitude for layer N.

        In overlap_2d mode this is interpolated from y_eff_raw_N(D) profile
        (microcanonical-windowed extraction). In legacy yukawa mode, return
        a deterministic fallback consistent with the configured B_N convention.
        """
        if N <= 0:
            return 0.0

        if self._b_mode_active == "overlap_2d" and self._b_overlap_profile is not None and N <= 3:
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
        prof = self._b_overlap_profile
        if self._b_mode_active == "overlap_2d" and prof is not None and "GUV" in prof:
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
        prof = self._b_overlap_profile
        if self._b_mode_active == "overlap_2d" and prof is not None and "LAMBDA123" in prof:
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
        c_uv = self.compute_ceh_uv(D=D, eta=eta, t_coh=t_coh, N_max=N_max)
        cfg = self._hll_uv_finite_match_config()
        c_match, meta = apply_ceh_finite_one_loop(c_tree=c_uv, cfg=cfg)
        return c_match, meta

    def hll_wilson_matrix_uv_match(self, D: float, eta: float, t_coh: float, N_max: int = 20) -> np.ndarray:
        c_match, _ = self.hll_wilson_matrix_uv_match_with_meta(D=D, eta=eta, t_coh=t_coh, N_max=N_max)
        return c_match

    def run_ceh_llrg(self, c_uv: np.ndarray, m2: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
        """
        Run leading-log RGE from UV matching scale to the configured low scale.

        Returns:
          (C_low, metadata) where metadata contains mu_match, mu_low, log_ratio.
        """
        fin_cfg = self._hll_uv_finite_match_config()
        c_match, fin_meta = apply_ceh_finite_one_loop(c_tree=c_uv, cfg=fin_cfg)
        cfg = self._hll_uv_rge_config()
        mu_match = mu_match_from_m2(m2, floor=cfg.floor)
        c_low, log_ratio = run_ceh_leading_log(c_match=c_match, mu_match=mu_match, cfg=cfg)
        meta = {
            "mu_match": float(mu_match),
            "mu_low": float(cfg.mu_low),
            "log_ratio": float(log_ratio),
            "kappa_diag": float(fin_meta["kappa_diag"]),
            "kappa_offdiag": float(fin_meta["kappa_offdiag"]),
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
        c_uv = self.compute_ceh_uv(D=D, eta=eta, t_coh=t_coh, N_max=N_max)
        m2 = self._hll_m2_vector(D)
        c_low, meta = self.run_ceh_llrg(c_uv=c_uv, m2=m2)
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
        c = self.hll_wilson_matrix_matched(D, eta, t_coh, N_max=N_max)
        c_ref = self.hll_wilson_matrix_matched(ref_D, ref_eta, t_coh, N_max=N_max)
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
        c = self.hll_wilson_matrix_uv_tree(D, eta, t_coh, N_max=N_max)
        c_ref = self.hll_wilson_matrix_uv_tree(ref_D, ref_eta, t_coh, N_max=N_max)
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
        c = self.hll_wilson_matrix_uv_rge(D, eta, t_coh, N_max=N_max)
        c_ref = self.hll_wilson_matrix_uv_rge(ref_D, ref_eta, t_coh, N_max=N_max)
        c_diag = np.diag(c)
        c_ref_diag = np.diag(c_ref)
        return total_width_ratio(c_diag=c_diag, c_diag_ref=c_ref_diag, cfg=cfg)

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
        amp = self.hll_channel_amplitude(layer_n, D, eta, t_coh, observable_mode=mode, N_max=nmax)
        amp_ref = self.hll_channel_amplitude(layer_n, ref_D, ref_eta, t_coh, observable_mode=mode, N_max=nmax)
        ratio = float(amp / max(amp_ref, 1e-30))
        if mode == "proxy_wratio":
            return ratio
        partial_ratio = ratio * ratio
        if mode == "eft_wilson_matched":
            width_ratio = self.hll_total_width_ratio_matched(
                D=D,
                eta=eta,
                t_coh=t_coh,
                ref_D=ref_D,
                ref_eta=ref_eta,
                N_max=nmax,
            )
            return float(partial_ratio / max(width_ratio, 1e-30))
        if mode == "eft_wilson_uv_tree":
            width_ratio = self.hll_total_width_ratio_uv_tree(
                D=D,
                eta=eta,
                t_coh=t_coh,
                ref_D=ref_D,
                ref_eta=ref_eta,
                N_max=nmax,
            )
            return float(partial_ratio / max(width_ratio, 1e-30))
        if mode == "eft_wilson_uv_rge":
            width_ratio = self.hll_total_width_ratio_uv_rge(
                D=D,
                eta=eta,
                t_coh=t_coh,
                ref_D=ref_D,
                ref_eta=ref_eta,
                N_max=nmax,
            )
            return float(partial_ratio / max(width_ratio, 1e-30))
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

        A1_eff, A2_eff = self._gamma_action_A12(D)
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
