#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "output" / "kinetic_action_chain" / "runtime_direct_detlin_d21_d64_centerwidth_probe_v1_3x3_samples.csv"
OUT = ROOT / "output" / "kinetic_action_chain" / "runtime_direct_detlin_d21_d64_centerwidth_probe_v1_3x3_contour.png"
PAPER = ROOT / "paper" / OUT.name


def main() -> None:
    df = pd.read_csv(CSV)
    centers = np.sort(df["g_center_d21"].unique())
    widths = np.sort(df["g_half_width_d21"].unique())
    z = np.empty((len(widths), len(centers)))
    for i, w in enumerate(widths):
        for j, c in enumerate(centers):
            z[i, j] = float(df[(df["g_center_d21"] == c) & (df["g_half_width_d21"] == w)]["p95_abs_delta_mu_mumu"].iloc[0])

    best = df.loc[df["p95_abs_delta_mu_mumu"].idxmin()]
    baseline = df[(df["g_center_d21"] == 0.06) & (df["g_half_width_d21"] == 0.04)].iloc[0]

    X, Y = np.meshgrid(centers, widths)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    levels = np.linspace(float(z.min()), float(z.max()), 12)
    cf = ax.contourf(X, Y, z, levels=levels, cmap="viridis_r")
    cs = ax.contour(X, Y, z, levels=levels, colors="white", linewidths=0.7, alpha=0.75)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    ax.scatter([baseline["g_center_d21"]], [baseline["g_half_width_d21"]], s=110, marker="o", color="#ffcc00", edgecolors="black", linewidths=0.9, label=f"v16 baseline ({baseline['p95_abs_delta_mu_mumu']:.4f})")
    ax.scatter([best["g_center_d21"]], [best["g_half_width_d21"]], s=120, marker="*", color="#ff6b6b", edgecolors="black", linewidths=0.9, label=f"best probe ({best['p95_abs_delta_mu_mumu']:.4f})")

    for row in df.itertuples(index=False):
        ax.text(float(row.g_center_d21) + 0.00035, float(row.g_half_width_d21) + 0.00025, f"{row.p95_abs_delta_mu_mumu:.3f}", fontsize=8, color="black")

    ax.set_title("D21 @ D=6.4 Exact 3x3 Center-Width Probe (fixed g_beta=0.31)")
    ax.set_xlabel("g_center_d21")
    ax.set_ylabel("g_half_width_d21")
    ax.legend(loc="upper left", frameon=True)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("p95 |Δμ_μμ|")

    note = (
        "Best probe shifts slightly to the upper-right: (0.070, 0.045).\n"
        f"Improvement vs v16 baseline: {float(best['p95_abs_delta_mu_mumu'] - baseline['p95_abs_delta_mu_mumu']):+.4f}"
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220)
    PAPER.write_bytes(OUT.read_bytes())
    plt.close(fig)


if __name__ == "__main__":
    main()
