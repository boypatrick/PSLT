#!/usr/bin/env python3
"""
Plot low-N comparison: g_N(Cardy) vs g_N(phase-space fp_2d).

Inputs:
  - output/gn_fp_impact/gn_profile_values.csv

Outputs:
  - output/gn_fp_impact/gn_cardy_vs_phase_space.csv
  - output/gn_fp_impact/gn_cardy_vs_phase_space.png
  - paper/gn_cardy_vs_phase_space.csv
  - paper/gn_cardy_vs_phase_space.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 12.5,
        "axes.titlesize": 16.5,
        "axes.labelsize": 14.5,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "legend.fontsize": 13.5,
    }
)


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "gn_fp_impact"
PAPER_DIR = ROOT / "paper"


def _nearest_row(df: pd.DataFrame, case: str, d_target: float) -> pd.Series:
    sub = df[df["case"] == case].copy()
    if sub.empty:
        raise KeyError(f"Missing case='{case}' in profile table.")
    idx = (sub["D"] - d_target).abs().idxmin()
    return sub.loc[idx]


def build_compare_table(df: pd.DataFrame, d_points: list[float], n_max_plot: int = 6) -> pd.DataFrame:
    rows = []
    cardy_case = "baseline_cardy"
    fp_case = "first_principles_fp_2d"
    for d in d_points:
        cardy = _nearest_row(df, cardy_case, d)
        fp = _nearest_row(df, fp_case, d)
        g3_cardy = float(cardy["g3"])
        g3_fp = float(fp["g3"])
        for n in range(1, n_max_plot + 1):
            g_cardy = float(cardy[f"g{n}"])
            g_fp = float(fp[f"g{n}"])
            rows.append(
                {
                    "D_target": float(d),
                    "D_used_cardy": float(cardy["D"]),
                    "D_used_fp2d": float(fp["D"]),
                    "N": int(n),
                    "g_cardy": g_cardy,
                    "g_fp2d": g_fp,
                    "g_cardy_over_g3": float(g_cardy / max(g3_cardy, 1e-30)),
                    "g_fp2d_over_g3": float(g_fp / max(g3_fp, 1e-30)),
                    "ratio_fp2d_to_cardy": float(g_fp / max(g_cardy, 1e-30)),
                }
            )
    return pd.DataFrame(rows)


def make_plot(tab: pd.DataFrame, out_png: Path) -> None:
    d_points = sorted(tab["D_target"].unique())
    fig, axes = plt.subplots(1, len(d_points), figsize=(14.8, 5.2), sharey=True)
    if len(d_points) == 1:
        axes = [axes]

    for ax, d in zip(axes, d_points):
        sub = tab[tab["D_target"] == d].sort_values("N")
        n = np.asarray(sub["N"], dtype=int)
        y_cardy = np.asarray(sub["g_cardy_over_g3"], dtype=float)
        y_fp = np.asarray(sub["g_fp2d_over_g3"], dtype=float)

        ax.plot(
            n,
            y_cardy,
            marker="o",
            markersize=7.8,
            markeredgewidth=0.8,
            color="#5b6770",
            linewidth=2.5,
            label="Cardy",
        )
        ax.plot(
            n,
            y_fp,
            marker="s",
            markersize=7.2,
            markeredgewidth=0.8,
            color="#1f77b4",
            linewidth=2.5,
            label="phase-space fp_2d",
        )
        ax.set_title(f"D = {int(d)}")
        ax.set_xticks(n)
        ax.set_xlabel("Layer N")
        ax.set_yscale("log")
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.8, which="both")
        if len(n) >= 3:
            # Emphasize low-N region used by generation interpretation.
            ax.axvspan(0.8, 3.2, color="#fef3c7", alpha=0.45, zorder=0)

    axes[0].set_ylabel(r"normalized $g_N/g_3$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Low-N comparison: Cardy baseline vs 2D phase-space profile", fontsize=23, y=1.09)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    in_csv = OUTDIR / "gn_profile_values.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input: {in_csv}. Run scan_gn_profile_impact.py first.")

    df = pd.read_csv(in_csv)
    tab = build_compare_table(df, d_points=[6.0, 12.0, 18.0], n_max_plot=6)

    out_csv = OUTDIR / "gn_cardy_vs_phase_space.csv"
    out_png = OUTDIR / "gn_cardy_vs_phase_space.png"
    tab.to_csv(out_csv, index=False)
    make_plot(tab, out_png)

    paper_csv = PAPER_DIR / out_csv.name
    paper_png = PAPER_DIR / out_png.name
    paper_csv.write_text(out_csv.read_text())
    paper_png.write_bytes(out_png.read_bytes())

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_png}")
    print(f"[saved] {paper_csv}")
    print(f"[saved] {paper_png}")


if __name__ == "__main__":
    main()
