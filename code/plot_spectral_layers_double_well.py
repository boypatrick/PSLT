#!/usr/bin/env python3
"""Generate the spectral-layer double-well topology intuition figure.

This is a schematic, geometry-facing figure for the Rank-2 kinetics section.
It shows how left/right localized lobes and a WKB bridge motivate the
two-dimensional kinetic subspace, without introducing a new numerical
certificate or model family.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "spectral_layers_double_well.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=float, default=8.0, help="schematic center separation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output image path")
    return parser.parse_args()


def schematic_double_well(z: np.ndarray, D: float) -> np.ndarray:
    """Smooth two-well profile with a central forbidden bridge."""
    sigma_well = 0.92
    sigma_barrier = 2.25
    return (
        0.22 * np.exp(-(z / sigma_barrier) ** 2)
        - 1.12 * np.exp(-((z + D / 2.0) ** 2) / (2.0 * sigma_well**2))
        - 1.12 * np.exp(-((z - D / 2.0) ** 2) / (2.0 * sigma_well**2))
    )


def level_segments(z: np.ndarray, U: np.ndarray, energy: float) -> list[tuple[float, float]]:
    mask = U < energy
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    chunks = np.split(idx, breaks + 1)
    return [(float(z[c[0]]), float(z[c[-1]])) for c in chunks]


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def turning_points(z: np.ndarray, U: np.ndarray, energy: float) -> list[float]:
    f = U - energy
    roots: list[float] = []
    for i in range(len(z) - 1):
        if f[i] == 0 or f[i] * f[i + 1] < 0:
            t = abs(f[i]) / (abs(f[i]) + abs(f[i + 1]))
            roots.append(float(z[i] * (1.0 - t) + z[i + 1] * t))
    return roots


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pslt_mpl_") as mpl_dir:
        os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
        os.environ.setdefault("XDG_CACHE_HOME", mpl_dir)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        z = np.linspace(-8.5, 8.5, 2400)
        U = schematic_double_well(z, args.D)

        levels = [
            (-0.86, r"$N=1$", "#1d4ed8"),
            (-0.61, r"$N=2$", "#7c3aed"),
            (-0.34, r"$N=3$", "#0f766e"),
        ]
        high_levels = [(-0.13, "#94a3b8"), (-0.055, "#cbd5e1")]

        fig, ax = plt.subplots(figsize=(13.2, 6.7), dpi=220)
        fig.subplots_adjust(left=0.07, right=0.985, top=0.89, bottom=0.14)

        ax.plot(z, U, color="#0f172a", lw=2.7, label=r"$V_{\rm eff}(z)-m_0^2$")
        ax.fill_between(z, U, 0.0, where=(U < 0.0), color="#dbeafe", alpha=0.55, interpolate=True)
        ax.fill_between(z, U, 0.0, where=(U > 0.0), color="#fee2e2", alpha=0.45, interpolate=True)
        ax.axhline(0.0, color="#64748b", lw=1.5, ls="--")
        ax.text(7.75, 0.025, "continuum threshold", ha="right", va="bottom", fontsize=10.5, color="#475569")

        for z0, label, color in [(-args.D / 2.0, r"$p_-$", "#7c3aed"), (args.D / 2.0, r"$p_+$", "#0f766e")]:
            ax.axvline(z0, color=color, lw=1.2, ls=":", alpha=0.9)
            ax.text(z0, -1.22, label, ha="center", va="bottom", fontsize=13, color=color, weight="bold")

        for energy, label, color in levels:
            segs = level_segments(z, U, energy)
            for x0, x1 in segs:
                ax.hlines(energy, x0, x1, color=color, lw=3.2)
                ax.plot([x0, x1], [energy, energy], "o", ms=3.2, color=color)
            if segs:
                ax.text(segs[-1][1] + 0.16, energy + 0.008, label, color=color, fontsize=12, va="center", weight="bold")

        for energy, color in high_levels:
            ax.hlines(energy, -6.8, 6.8, color=color, lw=2.0, ls=(0, (6, 4)), alpha=0.75)
        ax.text(5.95, -0.105, "higher layers\nfade toward continuum", color="#64748b", fontsize=10.2, ha="left", va="center")
        ax.annotate(
            "",
            xy=(7.05, -0.005),
            xytext=(6.30, -0.10),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#94a3b8", alpha=0.9),
        )

        energy_wkb = levels[-1][0]
        roots = turning_points(z, U, energy_wkb)
        if len(roots) >= 4:
            x_left, x_right = roots[1], roots[2]
        else:
            x_left, x_right = -2.0, 2.0
        bridge_x = np.linspace(x_left, x_right, 240)
        bridge_y = energy_wkb + 0.035 + 0.13 * np.sin(np.pi * (bridge_x - x_left) / (x_right - x_left))
        ax.plot(bridge_x, bridge_y, color="#f97316", lw=2.4, ls=(0, (4, 3)))
        ax.add_patch(
            FancyArrowPatch(
                (bridge_x[80], bridge_y[80]),
                (bridge_x[170], bridge_y[170]),
                arrowstyle="-|>",
                mutation_scale=13,
                lw=0,
                color="#f97316",
            )
        )
        bridge_idx = np.clip(np.searchsorted(z, bridge_x), 0, len(z) - 1)
        ax.fill_between(bridge_x, energy_wkb, U[bridge_idx], where=(U[bridge_idx] > energy_wkb), color="#fed7aa", alpha=0.42)
        ax.text(0.0, energy_wkb + 0.19, r"WKB bridge $\mathcal{B}_N$", color="#ea580c", fontsize=11.5, ha="center", weight="bold")

        base = -0.98
        amplitude = 0.18
        psi_left = base + amplitude * gaussian(z, -args.D / 2.0, 0.72)
        psi_right = base + amplitude * gaussian(z, args.D / 2.0, 0.72)
        mask_left = np.abs(z + args.D / 2.0) < 1.65
        mask_right = np.abs(z - args.D / 2.0) < 1.65
        ax.plot(z[mask_left], psi_left[mask_left], color="#2563eb", lw=2.0)
        ax.fill_between(z[mask_left], base, psi_left[mask_left], color="#bfdbfe", alpha=0.72)
        ax.plot(z[mask_right], psi_right[mask_right], color="#14b8a6", lw=2.0)
        ax.fill_between(z[mask_right], base, psi_right[mask_right], color="#99f6e4", alpha=0.62)
        ax.text(-args.D / 2.0, base + amplitude + 0.035, r"$|N,L\rangle$", ha="center", fontsize=11.5, color="#1d4ed8")
        ax.text(args.D / 2.0, base + amplitude + 0.035, r"$|N,R\rangle$", ha="center", fontsize=11.5, color="#0f766e")
        ax.annotate(
            r"$\mathcal{C}_N=\mathrm{span}\{|N,L\rangle,|N,R\rangle\}$",
            xy=(0.0, base + 0.035),
            xytext=(0.0, -1.30),
            ha="center",
            fontsize=12.5,
            color="#111827",
            arrowprops=dict(arrowstyle="-[,widthB=7.6,lengthB=0.55", lw=1.2, color="#111827"),
        )

        formula = (
            r"$S_N=\int_{\mathcal{B}_N}\sqrt{(V_{\rm eff}-\omega_N^2)_+}\,ds$"
            + "\n"
            + r"$r_N=\eta e^{-2S_N}$"
        )
        ax.text(
            -8.05,
            0.205,
            formula,
            fontsize=11.1,
            color="#111827",
            va="top",
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#fdba74", alpha=0.96),
        )

        ax.set_title("Spectral layers in a double-well topology", fontsize=16, weight="bold", pad=14)
        ax.set_xlabel(r"projected coordinate $z$", fontsize=12)
        ax.set_ylabel("shifted effective potential / layer energy", fontsize=12)
        ax.set_xlim(-8.5, 8.5)
        ax.set_ylim(-1.36, 0.34)
        ax.grid(True, color="#e5e7eb", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10)

        fig.text(
            0.5,
            0.055,
            "Schematic topology: localized left/right lobes and WKB leakage produce the Rank-2 two-center kinetic subspace.",
            ha="center",
            fontsize=10,
            color="#475569",
        )

        fig.savefig(output, bbox_inches="tight", facecolor="white")
        print(f"wrote {output}")
        print(f"parameters D={args.D:g}")


if __name__ == "__main__":
    main()
