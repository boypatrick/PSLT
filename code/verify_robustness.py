#!/usr/bin/env python3
"""
Verify Robustness (verify_robustness.py)

Generates "Advanced Verification" plots for the PSLT Paper Hardening Phase.
Addresses reviewer requests for:
1. Convergence checks (Grid resolution, N_max truncation).
2. Parameter sensitivity analysis ("Tuning" critique).

Outputs:
- output/verification_convergence.png
- output/verification_sensitivity.png
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import consolidated library
sys.path.insert(0, str(Path(__file__).parent))
from pslt_lib import PSLTKinetics, PSLTParameters

OUTDIR = Path(__file__).parent.parent / "output"
OUTDIR.mkdir(exist_ok=True, parents=True)

def check_convergence():
    print("Running Convergence Checks...")
    
    # 1. N_max Convergence
    # How does the Generation Ratio R3 change as we include more high-N layers?
    # If physics is sound, it should saturate quickly (no runaway denominator).
    
    kinetics = PSLTKinetics()
    D_fixed, eta_fixed, t_coh = 10.0, 1.0, 5.0
    
    # Keep this light enough to run on a laptop/CI while still demonstrating
    # saturation of the observable (reviewer expectation: no high-N runaway).
    N_max_vals = [10, 20, 40, 80]
    R3_vals = []
    
    print("  Testing N_max truncation stability...")
    for nm in N_max_vals:
        _, P_N, meta = kinetics.get_probabilities(D_fixed, eta_fixed, t_coh, N_max=nm)
        R3_vals.append(meta["generation_ratio"])
        
    # 2. Grid Resolution Convergence
    # Check if R3 averaged over the phase space changes with grid size
    print("  Testing Phase Space Grid stability...")
    # Grid convergence: show that a global metric converges as resolution increases.
    # Use a modest set of resolutions to keep runtime bounded.
    resolutions = [20, 40, 60]
    mean_R3s = []
    
    D_range = np.linspace(4, 20, 100) # Reference fine axis
    eta_range = np.linspace(0.2, 4.0, 100)
    
    # We sample 'res' points from these ranges
    for res in resolutions:
        Ds = np.linspace(4, 20, res)
        etas = np.linspace(0.2, 4.0, res)
        # Monte Carlo-ish sum? Or direct grid average?
        # Direct grid average
        total_r3 = 0.0
        count = 0
        for e in etas:
            for d in Ds:
                _, _, meta = kinetics.get_probabilities(d, e, t_coh, N_max=10)
                total_r3 += meta["generation_ratio"]
                count += 1
        mean_R3s.append(total_r3 / count)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # N_max Plot
    axes[0].plot(N_max_vals, R3_vals, 'o-', color='navy', lw=2)
    axes[0].set_xlabel(r"Truncation $N_{\max}$")
    axes[0].set_ylabel(r"Generation Ratio $\mathcal{R}_3$")
    axes[0].set_title(r"Stability against High-N Truncation")
    axes[0].grid(True, alpha=0.3)
    # axes[0].axhline(y=R3_vals[-1], color='gray', linestyle='--')
    
    # Grid Plot
    axes[1].plot(resolutions, mean_R3s, 's-', color='darkgreen', lw=2)
    axes[1].set_xlabel(r"Grid Resolution ($N \times N$)")
    axes[1].set_ylabel(r"Mean Phase Space $\langle \mathcal{R}_3 \rangle$")
    axes[1].set_title(r"Grid Convergence of Global Metrics")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTDIR / "verification_convergence.png", dpi=150)
    plt.close()
    print("Saved: verification_convergence.png")

def check_sensitivity():
    print("Running Sensitivity Analysis...")
    
    # Base parameters
    defaults = PSLTParameters()
    base_kinetics = PSLTKinetics(defaults)
    D_fix, eta_fix, t_coh = 8.0, 1.0, 5.0
    
    _, _, base_meta = base_kinetics.get_probabilities(D_fix, eta_fix, t_coh)
    base_R3 = base_meta["generation_ratio"]
    
    # Parameters to perturb
    perturbations = [
        ("c_eff", 0.5, 0.1),  # +/- 10%
        ("nu", 5.0, 0.1),
        ("chi", 0.2, 0.1),
        ("t_coh", 5.0, 0.2)   # +/- 20% for time
    ]
    
    results = []
    labels = []
    
    print(f"  Base R3 at (D={D_fix}, eta={eta_fix}): {base_R3:.4f}")
    
    for param_name, base_val, frac in perturbations:
        # High pert
        p_high = PSLTParameters()
        if param_name != "t_coh":
            setattr(p_high, param_name, base_val * (1 + frac))
        k_high = PSLTKinetics(p_high)
        val_high = base_val * (1 + frac) if param_name == "t_coh" else t_coh
        
        _, _, meta_high = k_high.get_probabilities(D_fix, eta_fix, val_high if param_name=="t_coh" else t_coh)
        
        # Low pert
        p_low = PSLTParameters()
        if param_name != "t_coh":
            setattr(p_low, param_name, base_val * (1 - frac))
        k_low = PSLTKinetics(p_low)
        val_low = base_val * (1 - frac) if param_name == "t_coh" else t_coh
        
        _, _, meta_low = k_low.get_probabilities(D_fix, eta_fix, val_low if param_name=="t_coh" else t_coh)
        
        # Calculate Log Sensitivity: d(ln R3) / d(ln P) ~ (dR/R) / (dP/P)
        # Using centered difference
        dR = meta_high["generation_ratio"] - meta_low["generation_ratio"]
        dP = (base_val * (1+frac)) - (base_val * (1-frac))
        
        # Normalized sensitivity S = (dR/R) / (dP/P)
        # approx (R_high - R_low)/R_base / (2*frac)
        S = (dR / base_R3) / (2 * frac)
        results.append(S)
        labels.append(f"{param_name}")
        
        print(f"    {param_name}: Sensitivity = {S:.4f}")

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['tab:blue' if abs(x)<1 else 'tab:red' for x in results]
    ax.barh(labels, results, color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel(r"Normalized Sensitivity $S_X = \frac{\partial \ln \mathcal{R}_3}{\partial \ln X}$")
    ax.set_title(r"Parameter Sensitivity Analysis (at Gen-2/3 Boundary)")
    
    # Add safe zone bands
    ax.axvspan(-0.5, 0.5, color='green', alpha=0.1, label="Robust (|S| < 0.5)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTDIR / "verification_sensitivity.png", dpi=150)
    plt.close()
    print("Saved: verification_sensitivity.png")

if __name__ == "__main__":
    check_convergence()
    check_sensitivity()
