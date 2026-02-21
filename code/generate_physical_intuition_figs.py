#!/usr/bin/env python3
"""
Generate physical-intuition figures for the PSLT manuscript.

Outputs (under paper/):
  - pslt_physical_chain.png
  - pslt_mixing_channels.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"


def add_box(ax, xy, wh, title, body, fc="#f3f6fb", ec="#2f3b52"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.3,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + 0.02 * w, y + 0.72 * h, title, fontsize=10, weight="bold", va="top")
    ax.text(x + 0.02 * w, y + 0.58 * h, body, fontsize=8.7, va="top")


def draw_chain_figure(path: Path):
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top-level flow boxes.
    add_box(
        ax,
        (0.03, 0.62),
        (0.20, 0.28),
        "Geometry Input",
        r"$D,\eta,\Omega(\rho,z)$" + "\nDual-center conformal geometry",
        fc="#eaf3ff",
    )
    add_box(
        ax,
        (0.28, 0.62),
        (0.22, 0.28),
        "Action-Derived Spectrum",
        r"$V_{\rm eff}\rightarrow\{E_N,\omega_N,\psi_N\}$" + "\nOne geometry, one operator chain",
        fc="#edf7ef",
    )
    add_box(
        ax,
        (0.55, 0.62),
        (0.21, 0.28),
        "Rank-2 Kinetics",
        r"$\Gamma_N=\max(0,\lambda_+)$" + "\n" + r"$\epsilon_{\rm mix}=\chi\bar\Gamma_N$",
        fc="#fff2e8",
    )
    add_box(
        ax,
        (0.79, 0.62),
        (0.18, 0.28),
        "Observable Layering",
        r"$W_N=B_N g_N(1-e^{-\Gamma_N t_{\rm coh}})$" + "\n" + r"$P_N=W_N/\sum_K W_K$",
        fc="#f6ecfb",
    )

    # Module strip.
    add_box(
        ax,
        (0.22, 0.20),
        (0.18, 0.26),
        "Module 1: Entropy",
        r"$g_N \sim e^{2\pi\sqrt{c_{\rm eff}N/6}}$" + "\nRegulated by " + r"$\kappa_g$",
        fc="#eef4ff",
    )
    add_box(
        ax,
        (0.42, 0.20),
        (0.18, 0.26),
        "Module 2: Mixing",
        r"$\chi_N^{\rm(sym)}\approx0$" + "\n" + r"$\chi_N^{(LR)}\neq0$ from splitting",
        fc="#fff4ea",
    )
    add_box(
        ax,
        (0.62, 0.20),
        (0.18, 0.26),
        "Module 3: Visibility",
        r"$B_N\propto(\tilde Y_N/\tilde Y_3)^{p_B}$" + "\n" + r"$B_{N>3}=1$",
        fc="#f3eefb",
    )

    add_box(
        ax,
        (0.82, 0.20),
        (0.15, 0.26),
        "Benchmarks",
        r"$\mathcal{R}_3,\ \mu_{\mu\mu}$" + "\nPhase diagram & proxy fit",
        fc="#f7f7f7",
    )

    # Arrows top chain
    arrow_kw = dict(arrowstyle="->", lw=1.5, color="#2f3b52")
    ax.annotate("", xy=(0.28, 0.76), xytext=(0.23, 0.76), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.55, 0.76), xytext=(0.50, 0.76), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.79, 0.76), xytext=(0.76, 0.76), arrowprops=arrow_kw)

    # Arrows from spectrum to modules and back to observables.
    ax.annotate("", xy=(0.31, 0.46), xytext=(0.39, 0.62), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.51, 0.46), xytext=(0.51, 0.62), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.71, 0.46), xytext=(0.62, 0.62), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.86, 0.46), xytext=(0.86, 0.62), arrowprops=arrow_kw)

    ax.text(
        0.03,
        0.95,
        "PSLT Closed Chain: One Geometry -> Spectrum -> Mixing -> Probabilities -> Observables",
        fontsize=13,
        weight="bold",
        ha="left",
        color="#1d2533",
    )
    ax.text(
        0.03,
        0.03,
        r"Key point: parity-symmetric overlap channel is null, while localized splitting defines nonzero $\chi_N^{(LR)}(D)$.",
        fontsize=9.5,
        color="#3b4455",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=260)
    plt.close(fig)


def draw_mixing_contrast(path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    # Panel A: symmetry cancellation.
    ax = axes[0]
    x = np.linspace(-2.8, 2.8, 600)
    integrand = x * np.exp(-x * x)
    ax.axhline(0, color="#333", lw=1)
    ax.plot(x, integrand, color="#1f77b4", lw=2, label=r"$\psi_1\,\delta V\,\psi_2$")
    ax.fill_between(x, 0, integrand, where=(integrand > 0), color="#7fb3ff", alpha=0.5)
    ax.fill_between(x, 0, integrand, where=(integrand < 0), color="#ff9ea6", alpha=0.5)
    ax.set_title(r"Parity Basis: $\chi_N^{\rm(sym)}\approx 0$")
    ax.set_xlabel(r"z (parity axis)")
    ax.set_ylabel("Integrand")
    ax.text(-2.55, 0.29, r"$+$ area", color="#2359b8", fontsize=9)
    ax.text(1.75, -0.26, r"$-$ area", color="#ba2b3b", fontsize=9)
    ax.text(-2.55, -0.36, r"$\int \psi_1^*\delta V\psi_2\,d^3x\approx 0$", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    # Panel B: localized splitting.
    ax = axes[1]
    z = np.linspace(-3, 3, 500)
    V = 0.09 * (z ** 2 - 1.5) ** 2 - 0.18
    ax.plot(z, V, color="#444", lw=2)
    e1, e2 = -0.10, 0.06
    ax.hlines([e1, e2], xmin=-2.8, xmax=2.8, colors=["#2e7d32", "#c62828"], linestyles=["-", "-"], lw=1.7)
    ax.text(1.95, e1 + 0.01, r"$E_1$", color="#2e7d32", fontsize=9)
    ax.text(1.95, e2 + 0.01, r"$E_2$", color="#c62828", fontsize=9)
    ax.annotate(
        "",
        xy=(0.6, e2),
        xytext=(0.6, e1),
        arrowprops=dict(arrowstyle="<->", lw=1.3, color="#7a1fa2"),
    )
    ax.text(0.72, (e1 + e2) / 2.0, r"$\Delta E=E_2-E_1$", color="#7a1fa2", fontsize=9, va="center")
    ax.text(-2.7, 0.21, r"$M_{LR}^{(H)}=\Delta E/2$", fontsize=10, weight="bold")
    ax.text(-2.7, 0.13, r"$\chi_N^{(LR)}=|M_{LR}^{(H)}|/\bar{\Gamma}_N$", fontsize=10)
    ax.text(-2.65, -0.24, r"$\psi_L$", fontsize=10, color="#1b5e20")
    ax.text(2.35, -0.24, r"$\psi_R$", fontsize=10, color="#1b5e20")
    ax.set_title(r"Localized Basis: Nonzero $\chi_N^{(LR)}$")
    ax.set_xlabel("Dual-well coordinate")
    ax.set_ylabel(r"$V_{\rm eff}$ (schematic)")
    ax.set_ylim(-0.33, 0.3)
    ax.grid(alpha=0.25)

    fig.suptitle("Why the Mixing Channel Must Be Redefined", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=260)
    plt.close(fig)


def main():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    chain_path = PAPER_DIR / "pslt_physical_chain.png"
    mix_path = PAPER_DIR / "pslt_mixing_channels.png"
    draw_chain_figure(chain_path)
    draw_mixing_contrast(mix_path)
    print(f"[done] wrote {chain_path}")
    print(f"[done] wrote {mix_path}")


if __name__ == "__main__":
    main()
