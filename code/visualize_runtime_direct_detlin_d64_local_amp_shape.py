#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv"
OUT = ROOT / "output" / "kinetic_action_chain" / "runtime_direct_detlin_d64_local_amp_shape.png"
PAPER = ROOT / "paper" / OUT.name

CENTER = 6.4
HALF_WIDTH = 0.40
ALPHA = 0.05


def tri_activation(x: np.ndarray, center: float, half_width: float) -> np.ndarray:
    return np.maximum(0.0, 1.0 - np.abs(x - center) / half_width)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    PAPER.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MAP)
    d_grid = np.array(sorted(df["D"].unique()), dtype=float)
    d_near = d_grid[(d_grid >= 5.6) & (d_grid <= 7.0)]

    x = np.linspace(5.6, 7.0, 600)
    act_cont = tri_activation(x, CENTER, HALF_WIDTH)
    act_grid = tri_activation(d_near, CENTER, HALF_WIDTH)
    corr_grid = ALPHA * act_grid
    mult_grid = np.exp(corr_grid)

    plt.rcParams.update(
        {
            "figure.figsize": (12, 9),
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    fig, axes = plt.subplots(3, 1, constrained_layout=True)

    ax = axes[0]
    ax.plot(x, act_cont, color="#1f77b4", lw=2.5, label="continuous triangular activation")
    ax.scatter(d_near, act_grid, color="#d62728", s=45, zorder=3, label="D60 grid samples")
    ax.axvline(CENTER, color="#555", lw=1.2, ls="--")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(5.6, 7.0)
    ax.set_ylabel("activation")
    ax.set_title("D=6.4 Local Amplitude Term: Window Shape")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(x, ALPHA * act_cont, color="#2ca02c", lw=2.5, label=r"$\Delta$ exponent = $\alpha a(D)$")
    ax.scatter(d_near, corr_grid, color="#d62728", s=45, zorder=3)
    for xd, yd in zip(d_near, corr_grid):
        if yd > 0:
            ax.text(xd, yd + 0.002, f"{yd:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.0, color="#888", lw=1.0)
    ax.set_xlim(5.6, 7.0)
    ax.set_ylabel("added exponent")
    ax.set_title(r"Actual local correction used in `v13` with $\alpha=+0.05$")
    ax.legend(loc="upper left")

    ax = axes[2]
    ax.plot(x, np.exp(ALPHA * act_cont), color="#9467bd", lw=2.5, label=r"multiplicative factor $\exp(\alpha a(D))$")
    ax.scatter(d_near, mult_grid, color="#d62728", s=45, zorder=3)
    for xd, yd in zip(d_near, mult_grid):
        if yd > 1.0:
            ax.text(xd, yd + 0.002, f"{yd:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="#888", lw=1.0)
    ax.set_xlim(5.6, 7.0)
    ax.set_ylabel("multiplier")
    ax.set_xlabel("D")
    ax.set_title("What the triangle does to the width side")
    ax.legend(loc="upper left")

    fig.suptitle(
        "This is a central triangular bump, not a flat-top plateau or an M-shape",
        fontsize=16,
        y=1.02,
    )
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    PAPER.write_bytes(OUT.read_bytes())
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
