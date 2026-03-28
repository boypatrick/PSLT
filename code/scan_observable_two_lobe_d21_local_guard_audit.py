#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_prolate_hybrid_reviewer_counterfactual_audit import REGIONS, _subset_rows

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
MAP_DIR = ROOT / "output" / "hll_signal_strength"
STEM = "observable_two_lobe_d21_local_guard_audit"

CANDIDATE_MAPS = {
    "D60_hotspot": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_D60E21_runtime_20260328.csv",
    "D21_guard": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_D21E41_runtime_20260328.csv",
}

PEAK_GRID = (0.85, 0.90, 0.93, 0.95, 0.97, 0.99)
CENTER_GRID = (6.39, 6.395, 6.40, 6.405, 6.41)
WIDTH_GRID = (0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030)


def _region_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for region, cfg in REGIONS.items():
        full_df = _subset_rows(Path(cfg["full_map"]), cfg["D_targets"]).rename(
            columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"}
        )
        base_df = _subset_rows(Path(cfg["base_map"]), cfg["D_targets"]).rename(
            columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"}
        )
        cand_df = _subset_rows(CANDIDATE_MAPS[region], cfg["D_targets"]).rename(
            columns={"mu_mumu": "mu_runtime", "chi2_mumu": "chi2_runtime"}
        )
        tables[region] = (
            full_df.merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
            .merge(cand_df[["D", "eta", "mu_runtime", "chi2_runtime"]], on=["D", "eta"], how="inner")
            .sort_values(["D", "eta"])
            .reset_index(drop=True)
        )
    return tables


def _apply_local_guard(mu_base: np.ndarray, mu_runtime: np.ndarray, D: np.ndarray, peak: float, center_D: float, width_D: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = (D.astype(float) - float(center_D)) / max(float(width_D), 1.0e-12)
    gaussian = np.exp(-0.5 * z * z)
    residual_frac = np.clip(1.0 - float(peak) * gaussian, 0.0, 1.0)
    mu_guard = mu_base.astype(float) + (mu_runtime.astype(float) - mu_base.astype(float)) * residual_frac
    return mu_guard, gaussian.astype(float), residual_frac.astype(float)


def _evaluate_combo(combo: dict[str, float | int], region_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, float | int | str]] = []
    slice_rows: list[dict[str, float | int | str]] = []
    detail_rows: list[dict[str, float | int | str]] = []

    for region, merged in region_tables.items():
        mu_guard, gaussian, residual_frac = _apply_local_guard(
            mu_base=merged["mu_base"].to_numpy(dtype=float),
            mu_runtime=merged["mu_runtime"].to_numpy(dtype=float),
            D=merged["D"].to_numpy(dtype=float),
            peak=float(combo["peak"]),
            center_D=float(combo["center_D"]),
            width_D=float(combo["width_D"]),
        )
        chi2_guard = ((mu_guard - 1.4) / 0.4) ** 2
        abs_delta_base = np.abs(merged["mu_base"].to_numpy(dtype=float) - merged["mu_full"].to_numpy(dtype=float))
        abs_delta_runtime = np.abs(merged["mu_runtime"].to_numpy(dtype=float) - merged["mu_full"].to_numpy(dtype=float))
        abs_delta_guard = np.abs(mu_guard - merged["mu_full"].to_numpy(dtype=float))
        abs_improvement_vs_base = abs_delta_base - abs_delta_guard
        abs_improvement_vs_runtime = abs_delta_runtime - abs_delta_guard

        region_detail = pd.DataFrame(
            {
                **combo,
                "region": region,
                "D": merged["D"].to_numpy(dtype=float),
                "eta": merged["eta"].to_numpy(dtype=float),
                "gaussian": gaussian,
                "residual_frac": residual_frac,
                "mu_full": merged["mu_full"].to_numpy(dtype=float),
                "mu_base": merged["mu_base"].to_numpy(dtype=float),
                "mu_runtime": merged["mu_runtime"].to_numpy(dtype=float),
                "mu_guard": mu_guard,
                "chi2_base": merged["chi2_base"].to_numpy(dtype=float),
                "chi2_runtime": merged["chi2_runtime"].to_numpy(dtype=float),
                "chi2_guard": chi2_guard,
                "abs_delta_base": abs_delta_base,
                "abs_delta_runtime": abs_delta_runtime,
                "abs_delta_guard": abs_delta_guard,
                "abs_improvement_vs_base": abs_improvement_vs_base,
                "abs_improvement_vs_runtime": abs_improvement_vs_runtime,
                "guard_shift_vs_runtime": mu_guard - merged["mu_runtime"].to_numpy(dtype=float),
                "guard_shift_vs_base": mu_guard - merged["mu_base"].to_numpy(dtype=float),
                "improved_vs_base": (abs_delta_guard < abs_delta_base).astype(float),
                "improved_vs_runtime": (abs_delta_guard < abs_delta_runtime).astype(float),
                "acceptance_flip_vs_base": ((merged["chi2_base"].to_numpy(dtype=float) <= 4.0) ^ (chi2_guard <= 4.0)).astype(float),
            }
        )
        detail_rows.append(region_detail)

        summary_rows.append(
            {
                **combo,
                "region": region,
                "p95_abs_delta_base": float(np.percentile(abs_delta_base, 95.0)),
                "p95_abs_delta_runtime": float(np.percentile(abs_delta_runtime, 95.0)),
                "p95_abs_delta_guard": float(np.percentile(abs_delta_guard, 95.0)),
                "max_abs_delta_base": float(np.max(abs_delta_base)),
                "max_abs_delta_runtime": float(np.max(abs_delta_runtime)),
                "max_abs_delta_guard": float(np.max(abs_delta_guard)),
                "mean_abs_improvement_vs_base": float(np.mean(abs_improvement_vs_base)),
                "mean_abs_improvement_vs_runtime": float(np.mean(abs_improvement_vs_runtime)),
                "improved_fraction_vs_base": float(np.mean(region_detail["improved_vs_base"])),
                "improved_fraction_vs_runtime": float(np.mean(region_detail["improved_vs_runtime"])),
                "acceptance_flip_fraction": float(np.mean(region_detail["acceptance_flip_vs_base"])),
            }
        )

        for d_val, grp in region_detail.groupby("D", sort=True):
            slice_rows.append(
                {
                    **combo,
                    "region": region,
                    "D": float(d_val),
                    "slice_p95_abs_delta_base": float(np.percentile(grp["abs_delta_base"], 95.0)),
                    "slice_p95_abs_delta_runtime": float(np.percentile(grp["abs_delta_runtime"], 95.0)),
                    "slice_p95_abs_delta_guard": float(np.percentile(grp["abs_delta_guard"], 95.0)),
                    "slice_mean_abs_improvement_vs_base": float(np.mean(grp["abs_improvement_vs_base"])),
                    "slice_mean_abs_improvement_vs_runtime": float(np.mean(grp["abs_improvement_vs_runtime"])),
                    "slice_improved_fraction_vs_base": float(np.mean(grp["improved_vs_base"])),
                    "slice_improved_fraction_vs_runtime": float(np.mean(grp["improved_vs_runtime"])),
                }
            )

    return (
        pd.concat(detail_rows, ignore_index=True),
        pd.DataFrame(summary_rows),
        pd.DataFrame(slice_rows),
    )


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    top = summary.head(20)
    if top.empty:
        return
    x = np.arange(len(top))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2))
    axes[0].bar(x - 0.18, top["p95_abs_delta_guard_D60_hotspot"], width=0.36, label="D60 guard", color="#e15759")
    axes[0].bar(x + 0.18, top["p95_abs_delta_guard_D21_guard"], width=0.36, label="D21 guard", color="#4e79a7")
    axes[0].axhline(float(top["p95_abs_delta_base_D60_hotspot"].iloc[0]), ls="--", color="#e15759", alpha=0.5)
    axes[0].axhline(float(top["p95_abs_delta_base_D21_guard"].iloc[0]), ls="--", color="#4e79a7", alpha=0.5)
    axes[0].set_title("Local-guard p95 |Δμ|")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(top["combo_id"].astype(int), rotation=90, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(top["p95_abs_delta_guard_D60_hotspot"], top["p95_abs_delta_guard_D21_guard"], c=top["peak"], cmap="viridis", s=45)
    axes[1].axvline(float(top["p95_abs_delta_base_D60_hotspot"].iloc[0]), ls="--", color="#e15759", alpha=0.5)
    axes[1].axhline(float(top["p95_abs_delta_base_D21_guard"].iloc[0]), ls="--", color="#4e79a7", alpha=0.5)
    axes[1].set_xlabel("D60 p95 |Δμ|")
    axes[1].set_ylabel("D21 p95 |Δμ|")
    axes[1].set_title("Tradeoff vs current mainline")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    region_tables = _region_tables()
    detail_frames = []
    summary_frames = []
    slice_frames = []

    combos = [
        (peak, center_D, width_D)
        for peak in PEAK_GRID
        for center_D in CENTER_GRID
        for width_D in WIDTH_GRID
    ]
    for idx, (peak, center_D, width_D) in enumerate(combos, start=1):
        print(
            f"[combo {idx}/{len(combos)}] peak={peak:.3f} center={center_D:.4f} width={width_D:.4f}",
            flush=True,
        )
        combo = {"combo_id": idx, "peak": float(peak), "center_D": float(center_D), "width_D": float(width_D)}
        detail_df, summary_df, slices_df = _evaluate_combo(combo, region_tables)
        detail_frames.append(detail_df)
        summary_frames.append(summary_df)
        slice_frames.append(slices_df)

    detail = pd.concat(detail_frames, ignore_index=True)
    summary_long = pd.concat(summary_frames, ignore_index=True)
    slices = pd.concat(slice_frames, ignore_index=True)

    summary = (
        summary_long.pivot_table(
            index=["combo_id", "peak", "center_D", "width_D"],
            columns="region",
            values=[
                "p95_abs_delta_base",
                "p95_abs_delta_runtime",
                "p95_abs_delta_guard",
                "max_abs_delta_base",
                "max_abs_delta_runtime",
                "max_abs_delta_guard",
                "mean_abs_improvement_vs_base",
                "mean_abs_improvement_vs_runtime",
                "improved_fraction_vs_base",
                "improved_fraction_vs_runtime",
                "acceptance_flip_fraction",
            ],
        )
        .reset_index()
    )
    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary["joint_objective_vs_base"] = np.maximum(
        summary["p95_abs_delta_guard_D60_hotspot"], summary["p95_abs_delta_guard_D21_guard"]
    )
    summary["beats_mainline_both"] = (
        (summary["p95_abs_delta_guard_D60_hotspot"] < summary["p95_abs_delta_base_D60_hotspot"])
        & (summary["p95_abs_delta_guard_D21_guard"] < summary["p95_abs_delta_base_D21_guard"])
    ).astype(float)
    summary["beats_runtime_both"] = (
        (summary["p95_abs_delta_guard_D60_hotspot"] < summary["p95_abs_delta_runtime_D60_hotspot"])
        & (summary["p95_abs_delta_guard_D21_guard"] < summary["p95_abs_delta_runtime_D21_guard"])
    ).astype(float)
    summary = summary.sort_values(
        [
            "joint_objective_vs_base",
            "p95_abs_delta_guard_D60_hotspot",
            "p95_abs_delta_guard_D21_guard",
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
                "candidate_maps": {k: str(v) for k, v in CANDIDATE_MAPS.items()},
                "scan": {
                    "peak": list(PEAK_GRID),
                    "center_D": list(CENTER_GRID),
                    "width_D": list(WIDTH_GRID),
                },
                "notes": (
                    "Counterfactual narrow local guard applied on top of the true runtime two-lobe mode. "
                    "The guard scales the extra runtime shift back toward the mainline only in a very narrow band around D=6.4. "
                    "This tests whether the runtime failure is a repairable local overshoot rather than a global two-lobe mechanism failure."
                ),
            },
            indent=2,
        )
    )

    for path in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(summary.head(20).to_string(index=False))
    print(f"\nWrote detail:  {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")


if __name__ == "__main__":
    main()
