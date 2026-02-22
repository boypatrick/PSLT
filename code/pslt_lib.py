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

    # Micro-degeneracy (Cardy-controlled envelope + high-N suppression)
    c_eff: float = 0.5      # Effective central charge (dimensionless)
    nu: float = 5.0         # Polynomial suppression exponent (dimensionless)
    kappa_g: float = 0.03    # High-N suppression strength in g_N: exp(-kappa_g*(N-1)^2)
    g_mode: str = "cardy"   # "cardy", "fp_1d", "fp_2d", "fp_1d_full", "fp_2d_full"
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
    chi: float = 0.2        # Rank-2 mixing parameter (dimensionless)
    chi_mode: str = "constant"  # "constant", "localized_interp", or "open_system"
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
    a0: float = 0.02        # Geometric perturbation strength (dimensionless)
    eps: float = 0.2        # Core regulator length [Length] ~ 1/[Mass] (scaled)
    
    # Visibility Scaling (Yukawa-proportional with compressed hierarchy)
    b_n_mode: str = "cumulative"  # "cumulative" or "single" over lepton Yukawas
    b_n_power: float = 0.30       # Sublinear compression: B_gen ∝ (y_gen)^{b_n_power}
    b_n_tail_mode: str = "saturate"  # "saturate" (paper baseline) or "gaussian"
    b_n_tail_beta: float = 0.50   # Used only when b_n_tail_mode == "gaussian"

    def __post_init__(self):
        if self.chi_mode not in {"constant", "localized_interp", "open_system"}:
            raise ValueError(f"Unsupported chi_mode='{self.chi_mode}'.")
        if self.chi_mode == "localized_interp":
            if len(self.chi_lr_D) < 2 or len(self.chi_lr_D) != len(self.chi_lr_vals):
                raise ValueError("chi_lr_D and chi_lr_vals must have equal length >=2 for localized_interp.")
        if self.chi_mode == "open_system":
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

        self._init_g_profiles()
        self._init_chi_profiles()

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
                g3 = self.g_N_cardy(3)
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

        entries: Dict[float, Dict[int, Tuple[float, float]]] = {}
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
            entries.setdefault(dval, {})[nval] = (lam, nps)

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
        for dval in d_sorted:
            lam_rows.append([entries[dval][int(n)][0] for n in mode_idx])
            nps_rows.append([entries[dval][int(n)][1] for n in mode_idx])

        return {
            "D": np.asarray(d_sorted, dtype=float),
            "mode_n": mode_idx,
            "lambda": np.asarray(lam_rows, dtype=float),
            "nps": np.asarray(nps_rows, dtype=float),
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

        if len(d_knots) == 1:
            lam = lam_knots[0].astype(float)
            nps = nps_knots[0].astype(float)
        else:
            lam = np.array([np.interp(D, d_knots, lam_knots[:, j]) for j in range(lam_knots.shape[1])], dtype=float)
            nps = np.array([np.interp(D, d_knots, nps_knots[:, j]) for j in range(nps_knots.shape[1])], dtype=float)
        out = {"lambda": np.maximum(lam, 1e-30), "nps": np.maximum(nps, 0.0)}
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

    def active_g_mode(self) -> str:
        return self._g_mode_active

    def _load_chi_open_profile(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
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
            gphi = row.get("gamma_phi_geom", row.get("gamma_phi", ""))
            gmix = row.get("gamma_mix_geom", row.get("gamma_mix", ""))
            delt = row.get("delta", "")
            gref = row.get("Gamma_ref", row.get("gamma_ref", ""))
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

    def _init_chi_profiles(self) -> None:
        mode = self.params.chi_mode
        self._chi_mode_active = mode
        self._chi_open_profile = None

        if mode != "open_system":
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
            path = self._auto_find_chi_open_csv()
        if path is not None:
            self._chi_open_profile = self._load_chi_open_profile(path)
            if self._chi_open_profile is None:
                print(f"Warning: could not parse open-system chi profile from {path}.")
        else:
            print("Warning: no chi_open_system profile file found for chi_mode=open_system.")

        if self._chi_open_profile is None:
            if len(self.params.chi_lr_D) >= 2 and len(self.params.chi_lr_D) == len(self.params.chi_lr_vals):
                self._chi_mode_active = "localized_interp"
                print("Warning: chi_mode=open_system requested but profile unavailable; falling back to localized_interp.")
            else:
                self._chi_mode_active = "constant"
                print("Warning: chi_mode=open_system requested but profile unavailable; falling back to constant chi.")

    def active_chi_mode(self) -> str:
        return self._chi_mode_active

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

    def g_N_effective(self, N: int, D: float) -> float:
        """
        Effective micro-degeneracy selector.

        Modes:
          - cardy: baseline surrogate.
          - fp_1d / fp_2d: first-principles N=1..3 shape correction (D-interpolated when
            available) blended onto cardy by g_fp_blend; N>3 follows cardy tail.
          - fp_1d_full / fp_2d_full: first-principles N=1..3 profile with
            full-profile continuation for N>3 (no Cardy tail fallback).
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
        g3_cardy = self.g_N_cardy(3)

        if mode in {"fp_1d_full", "fp_2d_full"}:
            # Anchor overall scale at N=3 baseline while replacing the full shape.
            if mode == "fp_2d_full":
                hat_full = self._build_fp_2d_full_hat_profile(D, g123_hat)
                if N <= len(hat_full):
                    return float(max(g3_cardy * float(hat_full[N - 1]), 1e-30))

                if len(hat_full) >= 2:
                    step_tail = float(hat_full[-1] / max(hat_full[-2], 1e-30))
                else:
                    step_tail = self.params.g_fp_full_tail_clip_min
                step_tail = float(np.clip(step_tail, self.params.g_fp_full_tail_clip_min, self.params.g_fp_full_tail_clip_max))
                ratio = float(hat_full[-1]) * (step_tail ** (N - len(hat_full)))
                return float(max(g3_cardy * ratio, 1e-30))

            if N <= 3:
                return float(max(g3_cardy * g123_hat[N - 1], 1e-30))
            # 1D full mode keeps the legacy geometric extension.
            r23 = float(g123_hat[1] / max(g123_hat[2], 1e-30))
            r13 = float(g123_hat[0] / max(g123_hat[2], 1e-30))
            r_tail = min(r23, r13)
            r_tail = float(np.clip(r_tail, self.params.g_fp_full_tail_clip_min, self.params.g_fp_full_tail_clip_max))
            return float(max(g3_cardy * (r_tail ** (N - 3)), 1e-30))

        if N <= 3:
            ratio_fp = float(g123_hat[N - 1])
            ratio_cardy = self.g_N_cardy(N) / max(g3_cardy, 1e-30)
            shape_corr = ratio_fp / max(ratio_cardy, 1e-30)
            blend = self.params.g_fp_blend
            return float(max(self.g_N_cardy(N) * (shape_corr ** blend), 1e-30))

        return self.g_N_cardy(N)

    # --- Visibility ---
    def B_N(self, N: int) -> float:
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

    def _interp_scalar(self, D: float, d_knots: np.ndarray, y_knots: np.ndarray) -> float:
        order = np.argsort(d_knots)
        d_sorted = d_knots[order]
        y_sorted = y_knots[order]
        return float(np.interp(D, d_sorted, y_sorted))

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
        - open_system: profile-interpolated (delta, gamma_phi, gamma_mix, gamma_ref)
          fed into two-level Lindblad dynamics, returning
          chi_eff = 2*gamma_mix*Cmax/gamma_ref.
        """
        mode = self._chi_mode_active
        if mode == "constant":
            return float(self.params.chi)

        if mode == "localized_interp":
            d_knots = np.asarray(self.params.chi_lr_D, dtype=float)
            chi_knots = np.asarray(self.params.chi_lr_vals, dtype=float)
            return self._interp_scalar(D, d_knots, chi_knots)

        if mode == "open_system" and self._chi_open_profile is not None:
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

        key = (int(N), float(round(D, 8)))
        pref = self._gamma_prefactor_cache.get(key, None)
        if pref is not None:
            return max(float(eta * pref), 0.0)

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

        g1 = gamma_sr(1, 1, self.params.A1)
        g2 = gamma_sr(2, 2, self.params.A2)

        # WKB tunneling suppression (dimensionless)
        w_val = self.omega_N(mu, N, D)
        S = self.action_S(mu, D, w_val)
        r_N = eta * np.exp(-2.0 * S)

        # Mixing term (shares units with g1,g2)
        chi_eff = self.chi_effective(D)
        eps_mix = chi_eff * math.sqrt(g1 * g2) if (g1 > 0 and g2 > 0) else 0.0

        # Largest eigenvalue of 2x2 symmetric matrix [[g1,eps],[eps,g2]]
        tr = g1 + g2
        disc = (g1 - g2) ** 2 + 4.0 * (eps_mix ** 2)
        lam_plus = 0.5 * (tr + math.sqrt(disc))

        pref = float(np.exp(-2.0 * S) * lam_plus)
        self._gamma_prefactor_cache[key] = pref
        Gamma = eta * pref
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
            g = self.g_N_effective(N, D)
            Gam = self.calculate_gamma_N(N, D, eta)
            B = self.B_N(N)
            
            # Main Closed Chain Equation
            factor = 1 - np.exp(-Gam * t_coh)
            w = B * g * factor
            
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
