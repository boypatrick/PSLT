#!/usr/bin/env python3
"""Generate the two-center conformal dumbbell intuition figure.

The figure is intentionally geometry-only: it visualizes the analytic
two-center conformal factor used in the worked example, without adding a new
model branch or numerical certificate.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "two_center_conformal_dumbbell.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--D", type=float, default=12.0, help="center separation")
    parser.add_argument("--a", type=float, default=1.0, help="source strength")
    parser.add_argument("--eps", type=float, default=0.35, help="Plummer regulator")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output image path",
    )
    return parser.parse_args()


def omega_two_center(rho: np.ndarray, z: np.ndarray, D: float, a: float, eps: float) -> np.ndarray:
    r_plus = np.sqrt(rho**2 + (z - D / 2.0) ** 2 + eps**2)
    r_minus = np.sqrt(rho**2 + (z + D / 2.0) ** 2 + eps**2)
    return 1.0 + a * (1.0 / r_plus + 1.0 / r_minus)


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    # Keep matplotlib cache writes out of the repository and user home.
    with tempfile.TemporaryDirectory(prefix="pslt_mpl_") as mpl_dir:
        os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
        os.environ.setdefault("XDG_CACHE_HOME", mpl_dir)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import LightSource

        z = np.linspace(-9.0, 9.0, 420)
        rho = np.linspace(-5.0, 5.0, 260)
        Z, RHO = np.meshgrid(z, rho)
        Omega = omega_two_center(RHO, Z, args.D, args.a, args.eps)

        # Compress the vertical range so the global dumbbell/neck geometry is
        # visible instead of being dominated by the two regulated source peaks.
        height = np.log(Omega)
        height = (height - height.min()) / (height.max() - height.min())

        fig = plt.figure(figsize=(12.2, 5.9), dpi=240)
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.28, 1.0],
            wspace=0.12,
            left=0.04,
            right=0.985,
            bottom=0.16,
            top=0.89,
        )

        ax = fig.add_subplot(grid[0, 0], projection="3d")
        light = LightSource(azdeg=300, altdeg=46)
        colors = light.shade(height, cmap=cm.viridis, vert_exag=0.75, blend_mode="soft")
        ax.plot_surface(
            Z,
            RHO,
            height,
            facecolors=colors,
            rstride=3,
            cstride=3,
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=0.98,
        )

        for z0, label in [(-args.D / 2.0, r"$p_-$"), (args.D / 2.0, r"$p_+$")]:
            iz = int(np.argmin(np.abs(z - z0)))
            ir = int(np.argmin(np.abs(rho)))
            h0 = float(height[ir, iz])
            ax.scatter([z0], [0.0], [h0 + 0.035], s=42, color="white", edgecolor="#111827", linewidth=0.7)
            ax.text(z0, 0.33, h0 + 0.105, label, color="#111827", fontsize=9, ha="center", weight="bold")

        ax.plot([-args.D / 2.0, args.D / 2.0], [0.0, 0.0], [1.07, 1.07], color="#1f2937", lw=1.2)
        ax.text(0.0, 0.20, 1.105, r"$D$", color="#1f2937", fontsize=9, ha="center", weight="bold")
        ax.text(0.0, -0.10, 0.24, "neck", color="#e0f2fe", fontsize=9, ha="center", weight="bold")
        ax.set_title(r"(a) Projected conformal landscape", pad=6, fontsize=10.5)
        ax.set_xlabel(r"$z$", labelpad=3, fontsize=9)
        ax.set_ylabel(r"$\rho$", labelpad=4, fontsize=9)
        ax.set_zticklabels([])
        ax.set_zlabel("")
        ax.tick_params(axis="both", which="major", labelsize=8, pad=0)
        ax.view_init(elev=30, azim=-58)
        ax.set_box_aspect((1.8, 0.95, 0.52))
        ax.set_xlim(-9.0, 9.0)
        ax.set_ylim(-5.0, 5.0)
        ax.set_zlim(0.0, 1.16)
        ax.grid(False)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.set_alpha(0.0)

        ax2 = fig.add_subplot(grid[0, 1])
        z_axis = np.linspace(-9.0, 9.0, 1400)
        omega_axis = omega_two_center(np.zeros_like(z_axis), z_axis, args.D, args.a, args.eps)
        profile = np.log(omega_axis)
        profile = (profile - profile.min()) / (profile.max() - profile.min())
        ax2.plot(z_axis, profile, color="#174ea6", lw=2.4)
        ax2.fill_between(z_axis, profile, 0.0, color="#cfe8f3", alpha=0.9)
        for z0, label, color in [(-args.D / 2.0, r"$p_-$", "#7c3aed"), (args.D / 2.0, r"$p_+$", "#0f766e")]:
            ax2.axvline(z0, color=color, lw=1.35, ls="--")
            ax2.text(z0, 1.035, label, color=color, ha="center", va="bottom", fontsize=11, weight="bold")
        ax2.annotate(
            "",
            xy=(-args.D / 2.0, 0.84),
            xytext=(args.D / 2.0, 0.84),
            arrowprops=dict(arrowstyle="<->", lw=1.4, color="#111827"),
        )
        ax2.text(0.0, 0.875, r"$D$", ha="center", va="bottom", fontsize=12, color="#111827")
        neck_idx = int(np.argmin(np.abs(z_axis)))
        ax2.annotate(
            "central neck",
            xy=(0.0, float(profile[neck_idx]) + 0.02),
            xytext=(0.0, 0.37),
            ha="center",
            fontsize=9.5,
            color="#0f172a",
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#0f172a"),
        )
        ax2.set_title(r"(b) On-axis profile", fontsize=10.5, pad=6)
        ax2.set_xlabel(r"projected axis $z$", fontsize=9)
        ax2.set_ylabel(r"compressed conformal height", fontsize=9)
        ax2.set_xlim(-9.0, 9.0)
        ax2.set_ylim(0.0, 1.10)
        ax2.tick_params(labelsize=8)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.grid(True, color="#e5e7eb", linewidth=0.75)

        fig.suptitle("Two-center conformal dumbbell", fontsize=14, y=0.965, weight="bold")
        fig.text(
            0.5,
            0.07,
            r"Geometry-only visualization of $\Omega(\rho,z;D)=1+a(r_+^{-1}+r_-^{-1})$, with height compressed as normalized $\log\Omega$.",
            ha="center",
            fontsize=9.0,
            color="#334155",
        )
        fig.text(
            0.5,
            0.035,
            r"The projected centers $p_-$ and $p_+$ generate a necked conformal landscape; the double-well potential is derived from this geometry.",
            ha="center",
            fontsize=8.6,
            color="#475569",
        )

        fig.savefig(output, bbox_inches="tight", facecolor="white")
        print(f"wrote {output}")
        print(f"parameters D={args.D:g} a={args.a:g} eps={args.eps:g}")


if __name__ == "__main__":
    main()
