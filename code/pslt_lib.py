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
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict

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
    
    # Geometry & Kinetics
    Omega_H: float = 0.9    # Horizon proxy angular velocity [Mass] (scaled by M)
    A1: float = 1.0         # l=1 amplitude (dimensionless prefactor for rate)
    A2: float = 1.0         # l=2 amplitude (dimensionless prefactor for rate)
    chi: float = 0.2        # Rank-2 mixing parameter (dimensionless)
    chi_mode: str = "constant"  # "constant" or "localized_interp"
    chi_lr_D: Tuple[float, ...] = (6.0, 12.0, 18.0)  # knots for localized chi(D)
    chi_lr_vals: Tuple[float, ...] = (4.01827e-4, 2.21414e-4, 2.13187e-4)  # chi_LR at knots
    a0: float = 0.02        # Geometric perturbation strength (dimensionless)
    eps: float = 0.2        # Core regulator length [Length] ~ 1/[Mass] (scaled)
    
    # Visibility Scaling (Yukawa-proportional with compressed hierarchy)
    b_n_mode: str = "cumulative"  # "cumulative" or "single" over lepton Yukawas
    b_n_power: float = 0.30       # Sublinear compression: B_gen ∝ (y_gen)^{b_n_power}
    b_n_tail_mode: str = "saturate"  # "saturate" (paper baseline) or "gaussian"
    b_n_tail_beta: float = 0.50   # Used only when b_n_tail_mode == "gaussian"

    def __post_init__(self):
        if self.chi_mode not in {"constant", "localized_interp"}:
            raise ValueError(f"Unsupported chi_mode='{self.chi_mode}'.")
        if self.chi_mode == "localized_interp":
            if len(self.chi_lr_D) < 2 or len(self.chi_lr_D) != len(self.chi_lr_vals):
                raise ValueError("chi_lr_D and chi_lr_vals must have equal length >=2 for localized_interp.")

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
        # Cache eta-independent kinetic prefactors keyed by (N, D_rounded).
        self._gamma_prefactor_cache: Dict[Tuple[int, float], float] = {}
        
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

    def chi_effective(self, D: float) -> float:
        """
        Effective mixing coefficient used in eps_mix.

        - constant: chi_eff = params.chi
        - localized_interp: piecewise-linear interpolation of chi_LR(D)
          with endpoint clamping outside knot range.
        """
        if self.params.chi_mode == "constant":
            return float(self.params.chi)

        d_knots = np.asarray(self.params.chi_lr_D, dtype=float)
        chi_knots = np.asarray(self.params.chi_lr_vals, dtype=float)
        order = np.argsort(d_knots)
        d_knots = d_knots[order]
        chi_knots = chi_knots[order]
        return float(np.interp(D, d_knots, chi_knots))

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
            g = self.g_N_cardy(N)
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
