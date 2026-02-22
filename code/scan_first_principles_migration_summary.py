#!/usr/bin/env python3
"""
Build a map-level comparison snapshot for first-principles migration.

Compared scenarios:
  1) baseline:        g_mode=fp_2d_full, chi_mode=localized_interp
  2) legacy reference g_mode=cardy, chi_mode=localized_interp
  3) chi open-system: g_mode=fp_2d_full, chi_mode=open_system

Inputs:
  - output/gn_fp_impact/gn_profile_impact.csv
  - output/chi_open_system/chi_open_system_sensitivity.csv

Outputs:
  - output/first_principles_migration/first_principles_migration_summary.csv
  - output/first_principles_migration/first_principles_migration_summary.png
  - paper/first_principles_migration_summary.csv
  - paper/first_principles_migration_summary.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "first_principles_migration"
PAPER_DIR = ROOT / "paper"

GN_CSV = ROOT / "output" / "gn_fp_impact" / "gn_profile_impact.csv"
CHI_CSV = ROOT / "output" / "chi_open_system" / "chi_open_system_sensitivity.csv"


def _pick_row(df: pd.DataFrame, case: str) -> pd.Series:
    sub = df[df["case"] == case]
    if sub.empty:
        raise KeyError(f"case='{case}' not found in {df.columns.tolist()}.")
    return sub.iloc[0]


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def build_summary() -> pd.DataFrame:
    if not GN_CSV.exists():
        raise FileNotFoundError(f"Missing input: {GN_CSV}")
    if not CHI_CSV.exists():
        raise FileNotFoundError(f"Missing input: {CHI_CSV}")

    gn_df = pd.read_csv(GN_CSV)
    chi_df = pd.read_csv(CHI_CSV)

    base = _pick_row(gn_df, "baseline_fp_2d_full")
    legacy = _pick_row(gn_df, "legacy_cardy")
    open_base = _pick_row(chi_df, "open_system_base")

    base_r3 = _to_float(base["f_R3_gt_0p90"])
    base_hmumu = _to_float(base["f_hmumu_chi2_le_4"])
    base_wgt3 = _to_float(base["f_winner_gt_3"])

    rows = [
        {
            "scenario": "baseline_fp2d_full_localized",
            "g_mode": "fp_2d_full",
            "chi_mode": "localized_interp",
            "f_R3_gt_0p90": base_r3,
            "delta_f_R3_gt_0p90_vs_baseline": 0.0,
            "f_hmumu_chi2_le_4": base_hmumu,
            "delta_f_hmumu_chi2_le_4_vs_baseline": 0.0,
            "f_winner_gt_3": base_wgt3,
            "delta_f_winner_gt_3_vs_baseline": 0.0,
            "best_chi2": _to_float(base["best_chi2"]),
            "mean_tail_prob": _to_float(base.get("mean_tail_prob", np.nan)),
            "chi_ratio_mean": 1.0,
            "chi_ratio_min": 1.0,
            "chi_ratio_max": 1.0,
        },
        {
            "scenario": "legacy_cardy_localized",
            "g_mode": "cardy",
            "chi_mode": "localized_interp",
            "f_R3_gt_0p90": _to_float(legacy["f_R3_gt_0p90"]),
            "delta_f_R3_gt_0p90_vs_baseline": _to_float(legacy["f_R3_gt_0p90"]) - base_r3,
            "f_hmumu_chi2_le_4": _to_float(legacy["f_hmumu_chi2_le_4"]),
            "delta_f_hmumu_chi2_le_4_vs_baseline": _to_float(legacy["f_hmumu_chi2_le_4"]) - base_hmumu,
            "f_winner_gt_3": _to_float(legacy["f_winner_gt_3"]),
            "delta_f_winner_gt_3_vs_baseline": _to_float(legacy["f_winner_gt_3"]) - base_wgt3,
            "best_chi2": _to_float(legacy["best_chi2"]),
            "mean_tail_prob": _to_float(legacy.get("mean_tail_prob", np.nan)),
            "chi_ratio_mean": 1.0,
            "chi_ratio_min": 1.0,
            "chi_ratio_max": 1.0,
        },
        {
            "scenario": "chi_open_system_base",
            "g_mode": "fp_2d_full",
            "chi_mode": "open_system",
            "f_R3_gt_0p90": _to_float(open_base["f_R3_gt_0p90"]),
            "delta_f_R3_gt_0p90_vs_baseline": _to_float(open_base["f_R3_gt_0p90"]) - base_r3,
            "f_hmumu_chi2_le_4": _to_float(open_base["f_hmumu_chi2_le_4"]),
            "delta_f_hmumu_chi2_le_4_vs_baseline": _to_float(open_base["f_hmumu_chi2_le_4"]) - base_hmumu,
            "f_winner_gt_3": _to_float(open_base["f_winner_gt_3"]),
            "delta_f_winner_gt_3_vs_baseline": _to_float(open_base["f_winner_gt_3"]) - base_wgt3,
            "best_chi2": _to_float(open_base["best_chi2"]),
            "mean_tail_prob": np.nan,
            "chi_ratio_mean": _to_float(open_base["ratio_mean"]),
            "chi_ratio_min": _to_float(open_base["ratio_min"]),
            "chi_ratio_max": _to_float(open_base["ratio_max"]),
        },
    ]
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, out_png: Path) -> None:
    labels = [
        "baseline\nfp_2d_full+localized",
        "legacy\ncardy+localized",
        "chi open\nfp_2d_full+open",
    ]
    x = np.arange(len(labels))
    colors = ["#5b6770", "#1f77b4", "#d97706"]

    r3 = np.asarray(df["f_R3_gt_0p90"], dtype=float)
    hm = np.asarray(df["f_hmumu_chi2_le_4"], dtype=float)
    w3 = np.asarray(df["f_winner_gt_3"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    metric_specs = [
        (r3, r"$f(\mathcal{R}_3>0.90)$", (0.0, 1.0)),
        (hm, r"$f(\chi^2_{\mu\mu}<4)$", (0.0, 1.0)),
        (w3, r"$f(N_{\rm win}>3)$", (0.0, max(0.02, float(np.max(w3) * 1.25 + 1e-6)))),
    ]

    for ax, (vals, title, ylim) in zip(axes, metric_specs):
        bars = ax.bar(x, vals, color=colors, edgecolor="#1f2937", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
        for b, v in zip(bars, vals):
            ypad = 0.012 * (ylim[1] - ylim[0])
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + ypad,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1f2937",
            )

    drift = np.asarray(df["delta_f_R3_gt_0p90_vs_baseline"], dtype=float)
    fig.suptitle(
        "Map-Level Migration Snapshot: baseline vs first-principles switches\n"
        f"R3 drift (absolute): legacy_cardy={drift[1]:+.4f}, open_system={drift[2]:+.4f}",
        fontsize=12.5,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary()
    out_csv = OUTDIR / "first_principles_migration_summary.csv"
    out_png = OUTDIR / "first_principles_migration_summary.png"
    summary.to_csv(out_csv, index=False)
    make_plot(summary, out_png)

    paper_csv = PAPER_DIR / out_csv.name
    paper_png = PAPER_DIR / out_png.name
    paper_csv.write_text(out_csv.read_text())
    paper_png.write_bytes(out_png.read_bytes())

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_png}")
    print(f"[saved] {paper_csv}")
    print(f"[saved] {paper_png}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
