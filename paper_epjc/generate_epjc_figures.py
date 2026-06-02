#!/usr/bin/env python3
"""Generate EPJC-facing figures that differ from the PRD preparation track.

Most release figures are copied from ``paper_prd`` because they are generated
from the same canonical artifacts.  This helper regenerates only figures whose
reader-facing text needs EPJC/journal-neutral wording.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def make_master_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = {
        "background": (0.04, 0.60, 0.18, 0.22, "A1 fixed\ntwo-center\nbackground", "#e6f2ff"),
        "operator": (0.29, 0.60, 0.18, 0.22, "Spectral operator\n$[-\\nabla^2+V_{\\rm eff}]\\psi_N=\\omega_N^2\\psi_N$", "#edf7ed"),
        "wkb": (0.54, 0.60, 0.18, 0.22, "WKB + two lobes\n$S_N$, $r_N=\\eta e^{-2S_N}$", "#fff5df"),
        "kinetics": (0.79, 0.60, 0.17, 0.22, "Rank-2 kinetics\n$\\Gamma_N=\\lambda_+(M_N)$", "#f7e9ff"),
        "weight": (0.18, 0.38, 0.24, 0.16, "Effective weight\n$g_N$", "#f0f0f0"),
        "visibility": (0.18, 0.14, 0.24, 0.16, "Visibility\n$B_N$", "#f0f0f0"),
        "occupancy": (0.67, 0.19, 0.28, 0.28, "Occupancy\n$P_N=W_N/\\sum_K W_K$", "#e9f9f7"),
    }

    def draw_box(item: tuple[float, float, float, float, str, str]) -> None:
        x, y, w, h, text, color = item
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.25,
            edgecolor="#243447",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    for item in boxes.values():
        draw_box(item)

    def anchor(name: str, side: str, dy: float = 0.0, dx: float = 0.0) -> tuple[float, float]:
        x, y, w, h, _, _ = boxes[name]
        if side == "left":
            return x + dx, y + h / 2 + dy
        if side == "right":
            return x + w + dx, y + h / 2 + dy
        if side == "top":
            return x + w / 2 + dx, y + h + dy
        if side == "bottom":
            return x + w / 2 + dx, y + dy
        raise ValueError(side)

    def arrow(start: tuple[float, float], end: tuple[float, float], rad: float = 0.0) -> None:
        ax.add_patch(FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.4,
            color="#243447",
            shrinkA=8,
            shrinkB=8,
            connectionstyle=f"arc3,rad={rad}",
        ))

    arrow(anchor("background", "right"), anchor("operator", "left"))
    arrow(anchor("operator", "right"), anchor("wkb", "left"))
    arrow(anchor("wkb", "right"), anchor("kinetics", "left"))
    arrow(anchor("kinetics", "bottom"), anchor("occupancy", "top", dx=0.06), rad=0.04)
    arrow(anchor("weight", "right"), anchor("occupancy", "left", dy=0.08))
    arrow(anchor("visibility", "right"), anchor("occupancy", "left", dy=-0.08))

    ax.text(
        0.5, 0.94,
        "Short-article claim boundary: conditional EFT-level occupancy closure",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5, 0.05,
        "Not a full EYMH proof of exactly three SM generations; H->mumu is reference-normalized diagnostic only.",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(OUT / "epjc_master_closure_diagram.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    make_master_diagram()
