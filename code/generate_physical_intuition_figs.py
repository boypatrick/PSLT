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
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background frame.
    frame = FancyBboxPatch(
        (0.02, 0.03),
        0.96,
        0.94,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#1f2a3a",
        facecolor="#f9fbfe",
        alpha=1.0,
    )
    ax.add_patch(frame)

    def stage_box(x, y, w, h, title, lines, fc, edge="#2a3448"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.015",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + 0.02 * w, y + h - 0.08 * h, title, fontsize=11.5, weight="bold", va="top", color="#1d2533")
        yy = y + h - 0.30 * h
        for line in lines:
            ax.text(x + 0.03 * w, yy, line, fontsize=9.6, va="top", color="#273246")
            yy -= 0.19 * h

    ax.text(
        0.04,
        0.95,
        "PSLT Action-Derived Closed Chain (Clean View)",
        fontsize=18,
        weight="bold",
        color="#1a2332",
        ha="left",
        va="top",
    )
    ax.text(
        0.04,
        0.91,
        "Geometry -> Operator -> Kinetics -> Layer Weights -> Observable Proxies",
        fontsize=11,
        color="#46556d",
        ha="left",
        va="top",
    )

    y_top, h_top = 0.60, 0.24
    w_top = 0.205
    xs = [0.045, 0.285, 0.525, 0.765]

    stage_box(
        xs[0], y_top, w_top, h_top,
        "1. Geometry Input",
        [
            r"$D,\ \eta,\ \Omega(\rho,z)$",
            "dual-center conformal background",
            "projection constraints fixed",
        ],
        fc="#e8f1ff",
    )
    stage_box(
        xs[1], y_top, w_top, h_top,
        "2. Action-Derived Operator",
        [
            r"$V_{\rm eff}\Rightarrow\{E_N,\omega_N,\psi_N\}$",
            "one solver chain for spectrum + WKB",
            "no extra ad-hoc barrier term",
        ],
        fc="#eaf8ef",
    )
    stage_box(
        xs[2], y_top, w_top, h_top,
        "3. Rank-2 Kinetics",
        [
            r"$\Gamma_N = \max(0,\lambda_+)$",
            r"$\epsilon_{\rm mix}=\chi_{\rm eff}\,\bar\Gamma_N$",
            r"$\chi^{\rm(sym)}\!\approx\!0,\ \chi^{(LR)}\!>\!0$",
        ],
        fc="#fff1e7",
    )
    stage_box(
        xs[3], y_top, w_top, h_top,
        "4. Observable Layering",
        [
            r"$W_N = B_N g_N(1-e^{-\Gamma_N t_{\rm coh}})$",
            r"$P_N = W_N / \sum_K W_K$",
            r"report: $\mathcal{R}_3,\ \mu_{\mu\mu},\ \mu_{ee},\ \mu_{\tau\tau}$",
        ],
        fc="#f5ecff",
    )

    arrow_kw = dict(arrowstyle="-|>", lw=1.8, color="#253449", mutation_scale=13)
    ax.annotate("", xy=(xs[1] - 0.012, y_top + 0.12), xytext=(xs[0] + w_top + 0.012, y_top + 0.12), arrowprops=arrow_kw)
    ax.annotate("", xy=(xs[2] - 0.012, y_top + 0.12), xytext=(xs[1] + w_top + 0.012, y_top + 0.12), arrowprops=arrow_kw)
    ax.annotate("", xy=(xs[3] - 0.012, y_top + 0.12), xytext=(xs[2] + w_top + 0.012, y_top + 0.12), arrowprops=arrow_kw)

    # Main equation panel.
    eq_patch = FancyBboxPatch(
        (0.12, 0.22),
        0.76,
        0.28,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#2f3b52",
        facecolor="#ffffff",
    )
    ax.add_patch(eq_patch)
    ax.text(0.50, 0.45, "Closed-Chain Equation", fontsize=15, weight="bold", ha="center", color="#1c2738")
    ax.text(
        0.50,
        0.365,
        r"$P_N(t_{\rm coh};D,\eta)=W_N/\left(\sum_K W_K\right)$",
        fontsize=16,
        ha="center",
        color="#1a2332",
    )
    ax.text(
        0.50,
        0.285,
        r"$W_N = B_N\,g_N\,\left(1-e^{-\Gamma_N t_{\rm coh}}\right)$",
        fontsize=15,
        ha="center",
        color="#1a2332",
    )

    # Connect top chain to equation panel.
    ax.annotate(
        "",
        xy=(0.50, 0.505),
        xytext=(0.50, y_top - 0.008),
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#1f2f45", mutation_scale=13),
    )
    ax.text(
        0.58,
        0.545,
        "aggregates",
        fontsize=9.0,
        color="#4a5870",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.3),
    )

    # Footer note.
    ax.text(
        0.04,
        0.09,
        r"Note: this figure is a readability-first schematic; detailed module benchmarks are listed in Appendix tables.",
        fontsize=9.8,
        color="#3d4b61",
        ha="left",
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
