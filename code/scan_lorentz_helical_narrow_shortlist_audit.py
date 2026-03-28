#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_lorentz_helical_counterfactual_audit import (
    BETA,
    INPUT_DETAIL,
    _evaluate_combo,
)


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
STEM = "lorentz_helical_narrow_shortlist_audit"

OMEGA_GRID = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
M_GRID = (4.0, 6.0, 8.0, 10.0, 12.0)
PHASE_GRID = tuple(float(np.pi / 4.0 + d) for d in (-0.35, -0.20, -0.10, 0.0, 0.10, 0.20, 0.35))
CENTER_GRID = (5.80, 5.90, 6.00, 6.10)
WIDTH_GRID = (0.03, 0.05, 0.07, 0.10)
ETA_COUPLE_GRID = (0.0, 0.15, 0.30)


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    top = summary.sort_values(
        ["joint_objective_vs_mainline", "p95_abs_delta_cf_D60_hotspot", "p95_abs_delta_cf_D21_guard"]
    ).head(20)
    if top.empty:
        return
    x = np.arange(len(top))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.2))
    axes[0].bar(x - 0.18, top["p95_abs_delta_cf_D60_hotspot"], width=0.36, label="D60 cf", color="#e15759")
    axes[0].bar(x + 0.18, top["p95_abs_delta_cf_D21_guard"], width=0.36, label="D21 cf", color="#4e79a7")
    axes[0].axhline(float(top["p95_abs_delta_mainline_D60_hotspot"].iloc[0]), ls="--", color="#e15759", alpha=0.5)
    axes[0].axhline(float(top["p95_abs_delta_mainline_D21_guard"].iloc[0]), ls="--", color="#4e79a7", alpha=0.5)
    axes[0].set_title("Top shortlist p95 |Δμ|")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(top["combo_id"].astype(int), rotation=90, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(top["p95_abs_delta_cf_D60_hotspot"], top["p95_abs_delta_cf_D21_guard"], c=top["omega"], cmap="viridis", s=45)
    axes[1].axvline(float(top["p95_abs_delta_mainline_D60_hotspot"].iloc[0]), ls="--", color="#e15759", alpha=0.5)
    axes[1].axhline(float(top["p95_abs_delta_mainline_D21_guard"].iloc[0]), ls="--", color="#4e79a7", alpha=0.5)
    axes[1].set_xlabel("D60 p95 |Δμ|")
    axes[1].set_ylabel("D21 p95 |Δμ|")
    axes[1].set_title("Top shortlist tradeoff")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    base_detail = pd.read_csv(INPUT_DETAIL)
    detail_frames = []
    summary_frames = []
    slice_frames = []

    combos = [
        (omega, m, phase, center_D, width_D, eta_couple)
        for omega in OMEGA_GRID
        for m in M_GRID
        for phase in PHASE_GRID
        for center_D in CENTER_GRID
        for width_D in WIDTH_GRID
        for eta_couple in ETA_COUPLE_GRID
    ]
    for idx, (omega, m, phase, center_D, width_D, eta_couple) in enumerate(combos, start=1):
        print(
            f"[combo {idx}/{len(combos)}] omega={omega:.2f} m={m:.2f} phase={phase:.3f} center={center_D:.2f} width={width_D:.2f} eta_couple={eta_couple:.2f}",
            flush=True,
        )
        combo = {
            "combo_id": idx,
            "beta": float(BETA),
            "omega": float(omega),
            "m": float(m),
            "phase": float(phase),
            "center_D": float(center_D),
            "width_D": float(width_D),
            "eta_couple": float(eta_couple),
        }
        detail_df, summary_df, slices_df = _evaluate_combo(combo, base_detail)
        detail_frames.append(detail_df)
        summary_frames.append(summary_df)
        slice_frames.append(slices_df)

    detail = pd.concat(detail_frames, ignore_index=True)
    summary_long = pd.concat(summary_frames, ignore_index=True)
    slices = pd.concat(slice_frames, ignore_index=True)

    summary = (
        summary_long.pivot_table(
            index=["combo_id", "beta", "omega", "m", "phase", "center_D", "width_D", "eta_couple"],
            columns="region",
            values=[
                "p95_abs_delta_mainline",
                "p95_abs_delta_lorentz",
                "p95_abs_delta_cf",
                "mean_abs_improvement_vs_mainline",
                "mean_abs_improvement_vs_lorentz",
                "improved_fraction_vs_mainline",
                "improved_fraction_vs_lorentz",
            ],
        )
        .reset_index()
    )
    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary["joint_objective_vs_mainline"] = np.maximum(
        summary["p95_abs_delta_cf_D60_hotspot"], summary["p95_abs_delta_cf_D21_guard"]
    )
    summary["d60_mainline_gain"] = summary["p95_abs_delta_mainline_D60_hotspot"] - summary["p95_abs_delta_cf_D60_hotspot"]
    summary["d21_mainline_gain"] = summary["p95_abs_delta_mainline_D21_guard"] - summary["p95_abs_delta_cf_D21_guard"]
    summary["beats_mainline_both"] = (
        (summary["p95_abs_delta_cf_D60_hotspot"] < summary["p95_abs_delta_mainline_D60_hotspot"])
        & (summary["p95_abs_delta_cf_D21_guard"] < summary["p95_abs_delta_mainline_D21_guard"])
    ).astype(float)
    summary["beats_lorentz_both"] = (
        (summary["p95_abs_delta_cf_D60_hotspot"] < summary["p95_abs_delta_lorentz_D60_hotspot"])
        & (summary["p95_abs_delta_cf_D21_guard"] < summary["p95_abs_delta_lorentz_D21_guard"])
    ).astype(float)
    summary = summary.sort_values(
        [
            "joint_objective_vs_mainline",
            "p95_abs_delta_cf_D60_hotspot",
            "p95_abs_delta_cf_D21_guard",
            "combo_id",
        ]
    ).reset_index(drop=True)

    detail_path = OUTDIR / f"{STEM}_detail.csv"
    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "input_detail": str(INPUT_DETAIL),
                "scan": {
                    "beta": BETA,
                    "omega": list(OMEGA_GRID),
                    "m": list(M_GRID),
                    "phase": list(PHASE_GRID),
                    "center_D": list(CENTER_GRID),
                    "width_D": list(WIDTH_GRID),
                    "eta_couple": list(ETA_COUPLE_GRID),
                },
                "notes": (
                    "Narrow shortlist around the promising Lorentz helical notch region. "
                    "The goal is to improve the D60 hotspot while keeping D21 guard below current mainline. "
                    "This focuses on phase ~ pi/4, moderate-to-high m, and narrow windows."
                ),
            },
            indent=2,
        )
    )

    for path in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(summary.head(15).to_string(index=False))
    print(f"\nWrote detail:  {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")


if __name__ == "__main__":
    main()
