#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_prolate_hybrid_reviewer_counterfactual_audit import REGIONS

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
MAP_DIR = ROOT / "output" / "hll_signal_strength"
STEM = "observable_two_lobe_d8compd724snapmaplate_runtime_validation"

BASE_CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost"
PARENT_CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate"
CANDIDATE_CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compd724snapmaplate"
DRIFT_TARGET_D = 7.24
TAG_SUFFIX = "20260331_d724snap_r1"

CASE_CONFIG = {
    "D60E21_fullmap": {
        "region_key": "D60_hotspot",
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21_refresh_20260324.csv",
        "ancestor": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D60E21_runtime_20260329_refsnapfix_d8maplate_r3.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D60E21_runtime_20260330_refsnapfix_d8comp_r1.csv",
    },
    "D21E41_fullmap": {
        "region_key": "D21_guard",
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41_refresh_20260324.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_currentbase_D21E41_20260329.csv",
        "ancestor": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D21E41_runtime_20260329_refsnapfix_d8maplate_r3.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D21E41_runtime_20260330_refsnapfix_d8comp_r1.csv",
    },
    "D60E41_holdout": {
        "region_key": "D60_hotspot",
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D60E41_20260330_crossgrid_holdout_r1.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E41_20260330_crossgrid_holdout_r1.csv",
        "ancestor": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D60E41_20260330_crossgrid_holdout_r1.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D60E41_20260330_crossgrid_holdout_r1.csv",
    },
    "D21E21_holdout": {
        "region_key": "D21_guard",
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D21E21_20260330_crossgrid_holdout_r1.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D21E21_20260330_crossgrid_holdout_r1.csv",
        "ancestor": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D21E21_20260330_crossgrid_holdout_r1.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D21E21_20260330_crossgrid_holdout_r1.csv",
    },
    "D80E41_monitor": {
        "region_key": "D60_hotspot",
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D80E41_20260330_d60_micro_monitor_r1.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D80E41_20260330_d60_micro_monitor_r1.csv",
        "ancestor": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D80E41_20260330_d60_micro_monitor_r1.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D80E41_20260330_d60_micro_monitor_r1.csv",
    },
    "D100E41_monitor": {
        "region_key": "D60_hotspot",
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D100E41_20260330_d60_micro_monitor_r1.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D100E41_20260330_d60_micro_monitor_r1.csv",
        "ancestor": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D100E41_20260330_d60_micro_monitor_r1.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D100E41_20260330_d60_micro_monitor_r1.csv",
    },
}


def _snap_targets_to_grid(unique_ds: np.ndarray, targets: list[float]) -> list[float]:
    snapped: list[float] = []
    for target in targets:
        idx = int(np.argmin(np.abs(unique_ds - float(target))))
        d_val = float(unique_ds[idx])
        if not any(np.isclose(d_val, prev, atol=1.0e-9) for prev in snapped):
            snapped.append(d_val)
    return snapped


def _load_case(case_name: str, cfg: dict[str, object], paths: dict[str, Path]) -> tuple[pd.DataFrame, list[float], float]:
    full_df = pd.read_csv(paths["full"]).rename(columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"})
    base_df = pd.read_csv(paths["base"]).rename(columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"})
    ancestor_df = pd.read_csv(paths["ancestor"]).rename(columns={"mu_mumu": "mu_ancestor", "chi2_mumu": "chi2_ancestor"})
    parent_df = pd.read_csv(paths["parent"]).rename(columns={"mu_mumu": "mu_parent", "chi2_mumu": "chi2_parent"})
    merged = (
        full_df[["D", "eta", "mu_full", "chi2_full"]]
        .merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
        .merge(ancestor_df[["D", "eta", "mu_ancestor", "chi2_ancestor"]], on=["D", "eta"], how="inner")
        .merge(parent_df[["D", "eta", "mu_parent", "chi2_parent"]], on=["D", "eta"], how="inner")
        .sort_values(["D", "eta"])
        .reset_index(drop=True)
    )
    unique_ds = np.sort(merged["D"].unique().astype(float))
    reviewer_ds = _snap_targets_to_grid(unique_ds, REGIONS[str(cfg["region_key"])]["D_targets"])
    drift_d = float(unique_ds[int(np.argmin(np.abs(unique_ds - DRIFT_TARGET_D)))])
    d_z = (merged["D"].to_numpy(dtype=float) - drift_d) / 0.005
    gaussian = np.exp(-0.5 * d_z * d_z)
    residual_frac = np.clip(1.0 - 1.0 * gaussian, 0.0, 1.0)
    floor = 1.0e-30
    merged["mu_cf"] = np.exp(
        (1.0 - residual_frac) * np.log(np.maximum(merged["mu_ancestor"].to_numpy(dtype=float), floor))
        + residual_frac * np.log(np.maximum(merged["mu_parent"].to_numpy(dtype=float), floor))
    )
    merged["chi2_cf"] = ((merged["mu_cf"] - 1.4) / 0.4) ** 2
    merged["case"] = case_name
    merged["abs_delta_base"] = (merged["mu_base"] - merged["mu_full"]).abs()
    merged["abs_delta_parent"] = (merged["mu_parent"] - merged["mu_full"]).abs()
    merged["abs_delta_cf"] = (merged["mu_cf"] - merged["mu_full"]).abs()
    merged["abs_improvement_vs_base"] = merged["abs_delta_base"] - merged["abs_delta_cf"]
    merged["abs_improvement_vs_parent"] = merged["abs_delta_parent"] - merged["abs_delta_cf"]
    merged["improved_vs_base"] = (merged["abs_delta_cf"] < merged["abs_delta_base"]).astype(float)
    merged["improved_vs_parent"] = (merged["abs_delta_cf"] < merged["abs_delta_parent"]).astype(float)
    merged["worsened_vs_parent"] = (merged["abs_delta_cf"] > merged["abs_delta_parent"]).astype(float)
    merged["acceptance_flip_vs_base"] = ((merged["chi2_base"] <= 4.0) ^ (merged["chi2_cf"] <= 4.0)).astype(float)
    reviewer_mask = np.zeros(len(merged), dtype=bool)
    dvals = merged["D"].to_numpy(dtype=float)
    for target in reviewer_ds:
        reviewer_mask |= np.isclose(dvals, target, atol=1.0e-9)
    merged["is_reviewer_target"] = reviewer_mask
    merged["is_drift_strip"] = np.isclose(dvals, drift_d, atol=1.0e-9)
    return merged, reviewer_ds, drift_d


def _subset_mask(df: pd.DataFrame, subset: str) -> pd.Series:
    if subset == "all_points":
        return pd.Series(True, index=df.index)
    if subset == "reviewer_targets":
        return df["is_reviewer_target"].astype(bool)
    if subset == "nonreviewer_complement":
        return ~df["is_reviewer_target"].astype(bool)
    if subset == "drift_strip":
        return df["is_drift_strip"].astype(bool)
    raise ValueError(subset)


def _summarize_subset(case_name: str, subset: str, df: pd.DataFrame, drift_d: float) -> dict[str, float | str | int]:
    worst_idx = df["abs_improvement_vs_parent"].idxmin()
    best_idx = df["abs_improvement_vs_parent"].idxmax()
    worst = df.loc[worst_idx]
    best = df.loc[best_idx]
    return {
        "case": case_name,
        "subset": subset,
        "drift_grid_D": float(drift_d),
        "count_points": int(len(df)),
        "p95_abs_delta_base": float(np.percentile(df["abs_delta_base"], 95.0)),
        "p95_abs_delta_parent": float(np.percentile(df["abs_delta_parent"], 95.0)),
        "p95_abs_delta_cf": float(np.percentile(df["abs_delta_cf"], 95.0)),
        "max_abs_delta_base": float(df["abs_delta_base"].max()),
        "max_abs_delta_parent": float(df["abs_delta_parent"].max()),
        "max_abs_delta_cf": float(df["abs_delta_cf"].max()),
        "mean_abs_improvement_vs_base": float(df["abs_improvement_vs_base"].mean()),
        "mean_abs_improvement_vs_parent": float(df["abs_improvement_vs_parent"].mean()),
        "improved_fraction_vs_base": float(df["improved_vs_base"].mean()),
        "improved_fraction_vs_parent": float(df["improved_vs_parent"].mean()),
        "worsened_fraction_vs_parent": float(df["worsened_vs_parent"].mean()),
        "acceptance_flip_fraction_vs_base": float(df["acceptance_flip_vs_base"].mean()),
        "best_point_D": float(best["D"]),
        "best_point_eta": float(best["eta"]),
        "best_point_improvement_vs_parent": float(best["abs_improvement_vs_parent"]),
        "worst_point_D": float(worst["D"]),
        "worst_point_eta": float(worst["eta"]),
        "worst_point_improvement_vs_parent": float(worst["abs_improvement_vs_parent"]),
    }


def _slice_rows(case_name: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d_val, grp in df.groupby("D", sort=True):
        rows.append(
            {
                "case": case_name,
                "D": float(d_val),
                "is_reviewer_target": float(grp["is_reviewer_target"].iloc[0]),
                "is_drift_strip": float(grp["is_drift_strip"].iloc[0]),
                "slice_p95_abs_delta_base": float(np.percentile(grp["abs_delta_base"], 95.0)),
                "slice_p95_abs_delta_parent": float(np.percentile(grp["abs_delta_parent"], 95.0)),
                "slice_p95_abs_delta_cf": float(np.percentile(grp["abs_delta_cf"], 95.0)),
                "slice_max_abs_delta_base": float(grp["abs_delta_base"].max()),
                "slice_max_abs_delta_parent": float(grp["abs_delta_parent"].max()),
                "slice_max_abs_delta_cf": float(grp["abs_delta_cf"].max()),
                "slice_mean_abs_improvement_vs_parent": float(grp["abs_improvement_vs_parent"].mean()),
                "slice_improved_fraction_vs_parent": float(grp["improved_vs_parent"].mean()),
                "slice_worsened_fraction_vs_parent": float(grp["worsened_vs_parent"].mean()),
                "slice_acceptance_flip_fraction_vs_base": float(grp["acceptance_flip_vs_base"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot(slices: pd.DataFrame, out_png: Path) -> None:
    order = [
        "D60E21_fullmap",
        "D60E41_holdout",
        "D80E41_monitor",
        "D100E41_monitor",
        "D21E41_fullmap",
        "D21E21_holdout",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)
    axes = axes.flatten()
    palette = {"base": "#4e79a7", "parent": "#e15759", "candidate": "#59a14f"}
    for ax, case_name in zip(axes, order):
        grp = slices.loc[slices["case"] == case_name].sort_values("D")
        if grp.empty:
            ax.axis("off")
            continue
        ax.plot(grp["D"], grp["slice_p95_abs_delta_base"], marker="o", color=palette["base"], label="base")
        ax.plot(grp["D"], grp["slice_p95_abs_delta_parent"], marker="o", color=palette["parent"], label="d8comp parent")
        ax.plot(grp["D"], grp["slice_p95_abs_delta_cf"], marker="o", color=palette["candidate"], label="d724snap")
        for d in grp.loc[grp["is_reviewer_target"] > 0.5, "D"].tolist():
            ax.axvline(d, color="k", ls=":", alpha=0.12)
        for d in grp.loc[grp["is_drift_strip"] > 0.5, "D"].tolist():
            ax.axvline(d, color="#f28e2b", ls="--", alpha=0.18)
        ax.set_title(case_name.replace("_", " "))
        ax.set_xlabel("D")
        ax.set_ylabel("slice p95 |Δμ|")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    path_table: dict[str, dict[str, Path]] = {}
    for case_name, cfg in CASE_CONFIG.items():
        path_table[case_name] = {
            "full": Path(str(cfg["full"])),
            "base": Path(str(cfg["base"])),
            "ancestor": Path(str(cfg["ancestor"])),
            "parent": Path(str(cfg["parent"])),
        }

    summary_rows = []
    slice_frames = []
    detail_frames = []
    for case_name, cfg in CASE_CONFIG.items():
        detail, reviewer_ds, drift_d = _load_case(case_name, cfg, path_table[case_name])
        detail_frames.append(detail)
        subsets = ["all_points", "reviewer_targets", "nonreviewer_complement"]
        if case_name.startswith(("D60", "D80", "D100")):
            subsets.append("drift_strip")
        for subset in subsets:
            sub = detail.loc[_subset_mask(detail, subset)].copy()
            summary_rows.append(_summarize_subset(case_name, subset, sub, drift_d))
        slice_frames.append(_slice_rows(case_name, detail))

    summary = pd.DataFrame(summary_rows)
    slices = pd.concat(slice_frames, ignore_index=True)
    detail = pd.concat(detail_frames, ignore_index=True).sort_values(["case", "D", "eta"]).reset_index(drop=True)

    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    detail_path = OUTDIR / f"{STEM}_detail.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    detail.to_csv(detail_path, index=False)
    _plot(slices, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "base_chain_mode": BASE_CHAIN_MODE,
                "parent_chain_mode": PARENT_CHAIN_MODE,
                "candidate_chain_mode": CANDIDATE_CHAIN_MODE,
                "cases": {
                    case_name: {key: (str(value) if isinstance(value, Path) else value) for key, value in cfg.items()}
                    for case_name, cfg in CASE_CONFIG.items()
                },
                "paths": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in path_table.items()},
                "drift_target_D": DRIFT_TARGET_D,
                "notes": (
                    "Formal validation for the grid-aware D≈7.24 snapped-center descendant using the exact "
                    "map-layer equivalent of the integrated runtime patch. The candidate starts from the "
                    "stored twolobe_d8compmaplate parent and applies a final mu_log blend back toward the "
                    "d8maplate ancestor around the active grid point nearest D≈7.24."
                ),
            },
            indent=2,
        )
    )

    for path in [summary_path, slices_path, detail_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(summary.to_string(index=False))
    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")
    print(f"Wrote detail:  {detail_path}")


if __name__ == "__main__":
    main()
