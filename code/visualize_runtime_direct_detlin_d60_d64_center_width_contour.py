#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPERDIR = ROOT / "paper"
OUT_TAG = os.environ.get("OUT_TAG", "d60_d64_sobol_doe_v3")
SAMPLES = OUTDIR / f"runtime_direct_detlin_{OUT_TAG}_samples.csv"
SUMMARY = OUTDIR / f"runtime_direct_detlin_{OUT_TAG}_summary.json"
OUTPNG = OUTDIR / f"runtime_direct_detlin_{OUT_TAG}_center_width_contour.png"


def main() -> None:
    samples = pd.read_csv(SAMPLES)
    with open(SUMMARY) as f:
        summary = json.load(f)

    current = summary["current_reference"]
    best_obs = summary["best_observed"]
    best_pred = summary["predicted_next_best"]

    x = samples["g_center"].to_numpy()
    y = samples["g_half_width"].to_numpy()
    z = samples["p95_abs_delta_mu_mumu"].to_numpy()
    tri = mtri.Triangulation(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    contour = ax.tricontourf(tri, z, levels=18, cmap="viridis_r")
    ax.tricontour(tri, z, levels=10, colors="white", linewidths=0.5, alpha=0.55)
    ax.scatter(x, y, c=z, cmap="viridis_r", s=42, edgecolor="black", linewidth=0.35)
    ax.scatter(
        [current["g_center"]],
        [current["g_half_width"]],
        marker="x",
        s=90,
        linewidth=2.0,
        color="#444444",
        label="v7 current",
    )
    ax.scatter(
        [best_obs["g_center"]],
        [best_obs["g_half_width"]],
        marker="*",
        s=180,
        color="#d62728",
        edgecolor="black",
        linewidth=0.5,
        label="v3 best observed",
    )
    ax.scatter(
        [best_pred["g_center"]],
        [best_pred["g_half_width"]],
        marker="D",
        s=70,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.4,
        label="v3 surrogate pred",
    )
    ax.set_xlabel("g_center")
    ax.set_ylabel("g_half_width")
    ax.set_title("Contour view of the center-width valley")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.colorbar(contour, ax=ax, label="p95 |Δmu_mumu|")

    ax = axes[1]
    best = samples.nsmallest(8, "p95_abs_delta_mu_mumu").copy()
    scatter = ax.scatter(
        samples["g_center"],
        samples["g_half_width"],
        c=samples["p95_abs_delta_mu_mumu"],
        cmap="viridis_r",
        s=40,
        alpha=0.35,
        edgecolor="none",
    )
    ax.plot(best["g_center"], best["g_half_width"], color="#d62728", linewidth=1.2, alpha=0.8)
    ax.scatter(
        best["g_center"],
        best["g_half_width"],
        color="#d62728",
        s=50,
        edgecolor="black",
        linewidth=0.35,
    )
    for _, row in best.iterrows():
        ax.annotate(
            f"{row['p95_abs_delta_mu_mumu']:.4f}",
            (row["g_center"], row["g_half_width"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#222222",
        )
    ax.set_xlabel("g_center")
    ax.set_ylabel("g_half_width")
    ax.set_title("Top observed points lie on a curved ridge/valley")
    fig.colorbar(scatter, ax=ax, label="p95 |Δmu_mumu|")

    fig.suptitle(
        "D60 D=6.4 local geometry around v7\n"
        "The low-residual region is a narrow center-width valley, not a single half-width point.",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUTPNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    (PAPERDIR / OUTPNG.name).write_bytes(OUTPNG.read_bytes())
    print(OUTPNG)


if __name__ == "__main__":
    main()
