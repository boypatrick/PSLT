#!/usr/bin/env python3
"""
Generate Plots for PSLT Paper (generate_plots.py)

This script reproduces all figures used in the PSLT verification paper.
It uses the unified `pslt_lib.py` library.

Outputs:
1. Three-generation phase diagram (Winner map & Generation Ratio)
2. H->mumu exclusion map and signal strength
3. Yukawa visibility hierarchy
4. Example layer probability distributions

Usage:
    python generate_plots.py

Output Directory:
    ../output/
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
import json
import csv
import sys
import traceback

# Import consolidated library
sys.path.insert(0, str(Path(__file__).parent))
from pslt_lib import PSLTKinetics, PSLTParameters, load_yukawa_data

OUTDIR = Path(__file__).parent.parent / "output"
OUTDIR.mkdir(exist_ok=True, parents=True)

# Paper-baseline parameters used across all figures in this script.
PAPER_BASELINE = {
    "c_eff": 0.5,
    "nu": 5.0,
    "kappa_g": 0.03,
    "g_mode": "fp_2d_full",
    "g_fp_full_window_blend": 0.8,
    "g_fp_full_tail_beta": 1.1,
    "g_fp_full_tail_shell_power": 0.0,
    "g_fp_full_tail_clip_min": 1e-3,
    "g_fp_full_tail_clip_max": 0.95,
    "chi_legacy": 0.2,
    "chi_mode": "localized_interp",
    "A1": 1.0,
    "A2": 1.0,
    "p_B": 0.30,
    "b_mode": "overlap_2d",
    "t_coh": 1.0,
    "hmumu_ref_D": 10.0,
    "hmumu_ref_eta": 1.0,
}

DEFAULT_CHI_D = np.array([6.0, 12.0, 18.0], dtype=float)
DEFAULT_CHI_VALS = np.array([4.01827e-4, 2.21414e-4, 2.13187e-4], dtype=float)
B_OVERLAP_CSV = OUTDIR / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"


def load_localized_chi_profile():
    """
    Load chi_LR(D) from Appendix-C extraction CSV (fine rows only).
    Fallback to default knots if file is unavailable.
    """
    path = OUTDIR / "chi_fp_2d" / "localized_chi_D6-12-18.csv"
    if not path.exists():
        print(f"Warning: {path} not found. Falling back to built-in chi(D) knots.")
        return DEFAULT_CHI_D.copy(), DEFAULT_CHI_VALS.copy()

    try:
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("level", "").strip().lower() == "fine":
                    rows.append((float(r["D"]), float(r["chi_LR"])))
        if len(rows) < 2:
            raise ValueError("not enough fine rows in localized chi csv")
        rows.sort(key=lambda t: t[0])
        dvals = np.array([t[0] for t in rows], dtype=float)
        chis = np.array([t[1] for t in rows], dtype=float)
        print(f"Loaded localized chi(D) profile from {path}")
        return dvals, chis
    except Exception as exc:
        print(f"Warning: failed to parse {path} ({exc}). Falling back to built-in chi(D) knots.")
        return DEFAULT_CHI_D.copy(), DEFAULT_CHI_VALS.copy()


def make_baseline_kinetics() -> PSLTKinetics:
    chi_d, chi_vals = load_localized_chi_profile()
    params = PSLTParameters(
        c_eff=PAPER_BASELINE["c_eff"],
        nu=PAPER_BASELINE["nu"],
        kappa_g=PAPER_BASELINE["kappa_g"],
        g_mode=PAPER_BASELINE["g_mode"],
        g_fp_full_window_blend=PAPER_BASELINE["g_fp_full_window_blend"],
        g_fp_full_tail_beta=PAPER_BASELINE["g_fp_full_tail_beta"],
        g_fp_full_tail_shell_power=PAPER_BASELINE["g_fp_full_tail_shell_power"],
        g_fp_full_tail_clip_min=PAPER_BASELINE["g_fp_full_tail_clip_min"],
        g_fp_full_tail_clip_max=PAPER_BASELINE["g_fp_full_tail_clip_max"],
        chi=PAPER_BASELINE["chi_legacy"],
        chi_mode=PAPER_BASELINE["chi_mode"],
        chi_lr_D=tuple(float(x) for x in chi_d),
        chi_lr_vals=tuple(float(x) for x in chi_vals),
        A1=PAPER_BASELINE["A1"],
        A2=PAPER_BASELINE["A2"],
        b_mode=PAPER_BASELINE["b_mode"],
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=PAPER_BASELINE["p_B"],
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
    )
    print(
        "Using baseline profile:",
        f"g_mode={params.g_mode},",
        "with mixing profile",
        f"mode={params.chi_mode},",
        f"knots={list(params.chi_lr_D)},",
        f"chi={list(params.chi_lr_vals)}"
    )
    return PSLTKinetics(params)

# =============================================================================
# Figure 1: Three-Generation Phase Diagram
# =============================================================================
def plot_phase_diagram():
    print("Generating Phase Diagrams...")
    kinetics = make_baseline_kinetics()
    
    # Scan parameters
    D_vals = np.linspace(4, 20, 60)
    eta_vals = np.linspace(0.2, 4.0, 60)
    t_coh = PAPER_BASELINE["t_coh"]
    N_max = 20
    
    winners = np.zeros((len(eta_vals), len(D_vals)))
    ratios = np.zeros((len(eta_vals), len(D_vals)))
    
    for i, eta in enumerate(eta_vals):
        for j, D in enumerate(D_vals):
            _, _, meta = kinetics.get_probabilities(D, eta, t_coh, N_max)
            winners[i, j] = meta["winner"]
            ratios[i, j] = meta["generation_ratio"]
            
    frac_95 = float(np.mean(ratios >= 0.95))
    frac_90 = float(np.mean(ratios >= 0.90))
    print(f"  Phase-space fraction with R3>=0.95: {frac_95:.3f}")
    print(f"  Phase-space fraction with R3>=0.90: {frac_90:.3f}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Winner Map
    cmap_gen = plt.get_cmap("tab20", N_max)
    bounds = np.arange(0.5, N_max + 1.5, 1.0)
    norm = BoundaryNorm(bounds, cmap_gen.N)
    
    im1 = axes[0].imshow(winners, origin='lower', aspect='auto', 
                         extent=[D_vals.min(), D_vals.max(), eta_vals.min(), eta_vals.max()],
                         cmap=cmap_gen, norm=norm)
    axes[0].set_title(r"Winner Layer $N^\star(D, \eta)$", fontsize=15)
    axes[0].set_xlabel("D (Dual-center separation)", fontsize=12)
    axes[0].set_ylabel(r"$\eta$ (Overlap)", fontsize=12)
    cbar1 = plt.colorbar(im1, ax=axes[0], ticks=range(1, N_max + 1))
    cbar1.set_label("Layer Index N", fontsize=12)
    
    # Add representative labels dynamically (avoid hard-coded generation assignment)
    for (d_anno, eta_anno) in [(6.0, 2.0), (15.0, 2.0)]:
        _, _, meta_anno = kinetics.get_probabilities(d_anno, eta_anno, t_coh, N_max)
        w = int(meta_anno["winner"])
        if w <= 3:
            label = f"Gen {w} (N={w})"
        else:
            label = f"Layer {w} (N={w})"
        axes[0].text(d_anno, eta_anno, label, color='white', ha='center', fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.5))

    # Right: Generation Ratio
    im2 = axes[1].imshow(ratios, origin='lower', aspect='auto',
                         extent=[D_vals.min(), D_vals.max(), eta_vals.min(), eta_vals.max()],
                         cmap='YlGn', vmin=0, vmax=1)
    axes[1].set_title(r"Generation Ratio $(P_1+P_2+P_3)/\Sigma P_N$", fontsize=15)
    axes[1].set_xlabel("D", fontsize=12)
    axes[1].set_ylabel(r"$\eta$", fontsize=12)
    plt.colorbar(im2, ax=axes[1], label="Ratio")
    
    # Contours
    cs = axes[1].contour(D_vals, eta_vals, ratios, levels=[0.9, 0.95], colors=['orange', 'lime'], linewidths=2)
    axes[1].clabel(cs, inline=True, fmt='%.2f')
    
    # Hatching
    axes[1].contourf(D_vals, eta_vals, ratios, levels=[0.95, 1.0], colors='none', hatches=['///'])

    plt.tight_layout()
    plt.savefig(OUTDIR / "three_generation_phase_diagram.png", dpi=200)
    plt.close()
    print("Saved: three_generation_phase_diagram.png")

# =============================================================================
# Figure 2: Example Probabilities (Bar Charts)
# =============================================================================
def plot_example_bars():
    print("Generating Example Bar Charts...")
    kinetics = make_baseline_kinetics()
    points = [
        (5.0, 2.0, "Example A (D=5)"),
        (8.0, 2.0, "Example B (D=8)"),
        (15.0, 2.0, "Example C (D=15)")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#8B8982', '#6B4E71', '#454545']
    
    for ax, (D, eta, title) in zip(axes, points):
        Ns, P_N, meta = kinetics.get_probabilities(D, eta, PAPER_BASELINE["t_coh"], N_max=20)
        
        bars = ax.bar(Ns, P_N, color=colors[:len(Ns)], edgecolor='black', alpha=0.8)
        # Highlight first 3
        for i in range(3):
            if i < len(bars):
                bars[i].set_edgecolor('gold')
                bars[i].set_linewidth(2.5)
                
        winner = meta.get('winner', None)
        r3 = meta.get('generation_ratio', 0.0)
        wlab = f"Winner N={winner}" if winner is not None else "Winner"
        ax.set_title(f"{title}\n{wlab}, R3={r3:.2%}")
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Layer N")
        if ax == axes[0]: ax.set_ylabel("Probability")
        
    plt.tight_layout()
    plt.savefig(OUTDIR / "three_generation_bars.png", dpi=200)
    plt.close()
    print("Saved: three_generation_bars.png")

# =============================================================================
# Figure 3: H->mumu Verification
# =============================================================================
def plot_hmumu_check():
    print("Generating H->mumu Plots...")

    # Load Observed Data
    try:
        data_path = Path(__file__).parent.parent / "data" / "pdg_leptons.json"
        with open(data_path) as f:
            d = json.load(f)["higgs_signal_strength"]["H_to_mumu"]
        mu_obs = d["value"]
        sigma_obs = d["combined_uncertainty"]
    except Exception:
        print("Warning: Could not load H->mumu data. Using default 1.4 +/- 0.4")
        mu_obs, sigma_obs = 1.4, 0.4

    kinetics = make_baseline_kinetics()

    # In PSLT, we use a minimal generation-2 proxy:
    #   W_2(D,eta) = B_2 g_2 [1 - exp(-Gamma_2 t_coh)]
    # and define the predicted signal strength as the ratio to a fixed reference point.
    t_coh = PAPER_BASELINE["t_coh"]

    def get_W2(D, eta):
        N = 2
        Gam = kinetics.calculate_gamma_N(N, D, eta)
        g = kinetics.g_N_effective(N, D)
        B = kinetics.B_N(N, D)
        return B * g * (1.0 - np.exp(-Gam * t_coh))

    # Reference point (fixed, not tuned)
    D0 = PAPER_BASELINE["hmumu_ref_D"]
    eta0 = PAPER_BASELINE["hmumu_ref_eta"]
    W2_ref = get_W2(D0, eta0)
    if W2_ref <= 0:
        raise RuntimeError("Reference W2_ref is non-positive; adjust (D0,eta0) or parameters.")

    # Scan (same region as the main phase diagram)
    D_vals = np.linspace(4, 20, 60)
    eta_vals = np.linspace(0.2, 4.0, 60)
    chi2 = np.zeros((len(eta_vals), len(D_vals)))
    mu_pred = np.zeros_like(chi2)

    for i, eta in enumerate(eta_vals):
        for j, D in enumerate(D_vals):
            val = get_W2(D, eta) / W2_ref
            mu_pred[i, j] = val
            chi2[i, j] = ((val - mu_obs) / sigma_obs) ** 2

    frac_proxy = float(np.mean(chi2 <= 4.0))
    min_idx = np.unravel_index(np.argmin(chi2), chi2.shape)
    best_chi2 = float(chi2[min_idx])
    best_point = (float(D_vals[min_idx[1]]), float(eta_vals[min_idx[0]]))
    print(f"  H->mumu fraction with chi2<=4.0 (proxy threshold): {frac_proxy:.3f}")
    print(f"  Best-fit (grid) chi2={best_chi2:.3f} at (D,eta)={best_point}")

    # Plot Chi2 (Exclusion map)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        chi2, origin="lower", aspect="auto",
        extent=[D_vals.min(), D_vals.max(), eta_vals.min(), eta_vals.max()],
        cmap="coolwarm", vmin=0, vmax=10
    )
    plt.colorbar(im, label=r"$\chi^2$")
    ax.contour(D_vals, eta_vals, chi2, levels=[1.0, 4.0], colors=["green", "orange"], linewidths=2)
    ax.set_title(r"H$\to\mu\mu$ Compatibility (proxy: $\chi^2<4$)")
    ax.set_xlabel("D")
    ax.set_ylabel(r"$\eta$")

    plt.tight_layout()
    plt.savefig(OUTDIR / "hmumu_exclusion.png", dpi=200)
    plt.close()

    # Plot Signal Strength
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        mu_pred, origin="lower", aspect="auto",
        extent=[D_vals.min(), D_vals.max(), eta_vals.min(), eta_vals.max()],
        cmap="viridis"
    )
    plt.colorbar(im, label=r"$\mu_{\rm pred}$")
    ax.contour(D_vals, eta_vals, mu_pred, levels=[mu_obs], colors="white", linestyles="--")
    ax.set_title(f"Predicted Signal Strength (Obs: {mu_obs} ± {sigma_obs})")
    ax.set_xlabel("D")
    ax.set_ylabel(r"$\eta$")

    plt.tight_layout()
    plt.savefig(OUTDIR / "hmumu_signal_strength.png", dpi=200)
    plt.close()
    print("Saved: hmumu_exclusion.png, hmumu_signal_strength.png")

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    try:
        print(f"Starting reproduction sequence in {OUTDIR}...")
        plot_phase_diagram()
        plot_example_bars()
        plot_hmumu_check()
        print("All plots generated successfully.")
    except Exception:
        traceback.print_exc()
        raise
