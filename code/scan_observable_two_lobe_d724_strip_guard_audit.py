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
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
MAP_DIR = ROOT / "output" / "hll_signal_strength"
STEM = "observable_two_lobe_d724_strip_guard_audit"

DRIFT_TARGET_D = 7.24
PLACEMENTS = ("mu_linear_blend", "mu_log_blend")
PEAK_GRID = (0.30, 0.50, 0.70, 0.85, 1.00)
CENTER_GRID = (7.232323, 7.240000, 7.248000, 7.254237)
WIDTH_GRID = (0.005, 0.010, 0.015, 0.020, 0.030)
KEY_D_TARGETS = (6.4, 7.2, 7.24, 8.0, 9.6)

CASE_CONFIG = {
    "D60E21_monitor": {
        "region_key": "D60_hotspot",
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D60E21_20260330_d60_micro_monitor_r1.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21_20260330_d60_micro_monitor_r1.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D60E21_20260330_d60_micro_monitor_r1.csv",
        "candidate": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D60E21_20260330_d60_micro_monitor_r1.csv",
    },
    "D60E41_monitor": {
        "region_key": "D60_hotspot",
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D60E41_20260330_d60_micro_monitor_r1.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E41_20260330_d60_micro_monitor_r1.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D60E41_20260330_d60_micro_monitor_r1.csv",
        "candidate": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D60E41_20260330_d60_micro_monitor_r1.csv",
    },
    "D80E41_monitor": {
        "region_key": "D60_hotspot",
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D80E41_20260330_d60_micro_monitor_r1.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D80E41_20260330_d60_micro_monitor_r1.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D80E41_20260330_d60_micro_monitor_r1.csv",
        "candidate": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D80E41_20260330_d60_micro_monitor_r1.csv",
    },
    "D100E41_monitor": {
        "region_key": "D60_hotspot",
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D100E41_20260330_d60_micro_monitor_r1.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D100E41_20260330_d60_micro_monitor_r1.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D100E41_20260330_d60_micro_monitor_r1.csv",
        "candidate": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D100E41_20260330_d60_micro_monitor_r1.csv",
    },
    "D21E41_canonical": {
        "region_key": "D21_guard",
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41_refresh_20260324.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_currentbase_D21E41_20260329.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D21E41_runtime_20260329_refsnapfix_d8maplate_r3.csv",
        "candidate": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D21E41_runtime_20260330_refsnapfix_d8comp_r1.csv",
    },
    "D21E21_holdout": {
        "region_key": "D21_guard",
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D21E21_20260330_crossgrid_holdout_r1.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D21E21_20260330_crossgrid_holdout_r1.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D21E21_20260330_crossgrid_holdout_r1.csv",
        "candidate": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D21E21_20260330_crossgrid_holdout_r1.csv",
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


def _load_case(case_name: str, cfg: dict[str, Path | str]) -> tuple[pd.DataFrame, list[float], float]:
    full_df = pd.read_csv(cfg["full"]).rename(columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"})
    base_df = pd.read_csv(cfg["base"]).rename(columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"})
    parent_df = pd.read_csv(cfg["parent"]).rename(columns={"mu_mumu": "mu_parent", "chi2_mumu": "chi2_parent"})
    cand_df = pd.read_csv(cfg["candidate"]).rename(columns={"mu_mumu": "mu_candidate", "chi2_mumu": "chi2_candidate"})
    merged = (
        full_df[["D", "eta", "mu_full", "chi2_full"]]
        .merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
        .merge(parent_df[["D", "eta", "mu_parent", "chi2_parent"]], on=["D", "eta"], how="inner")
        .merge(cand_df[["D", "eta", "mu_candidate", "chi2_candidate"]], on=["D", "eta"], how="inner")
        .sort_values(["D", "eta"])
        .reset_index(drop=True)
    )
    unique_ds = np.sort(merged["D"].unique().astype(float))
    reviewer_ds = _snap_targets_to_grid(unique_ds, REGIONS[str(cfg["region_key"])]["D_targets"])
    drift_d = float(unique_ds[int(np.argmin(np.abs(unique_ds - DRIFT_TARGET_D)))])
    merged["case"] = case_name
    merged["abs_delta_base"] = (merged["mu_base"] - merged["mu_full"]).abs()
    merged["abs_delta_parent"] = (merged["mu_parent"] - merged["mu_full"]).abs()
    merged["abs_delta_candidate"] = (merged["mu_candidate"] - merged["mu_full"]).abs()
    reviewer_mask = np.zeros(len(merged), dtype=bool)
    dvals = merged["D"].to_numpy(dtype=float)
    for target in reviewer_ds:
        reviewer_mask |= np.isclose(dvals, target, atol=1.0e-9)
    merged["is_reviewer_target"] = reviewer_mask
    merged["is_drift_strip"] = np.isclose(dvals, drift_d, atol=1.0e-9)
    return merged, reviewer_ds, drift_d


def _apply_guard(
    placement: str,
    mu_parent: np.ndarray,
    mu_candidate: np.ndarray,
    d_vals: np.ndarray,
    peak: float,
    center_d: float,
    width_d: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = (d_vals.astype(float) - float(center_d)) / max(float(width_d), 1.0e-12)
    gauss = np.exp(-0.5 * z * z)
    residual_frac = np.clip(1.0 - float(peak) * gauss, 0.0, 1.0)
    floor = 1.0e-30
    if placement == "mu_linear_blend":
        mu_guard = mu_parent + (mu_candidate - mu_parent) * residual_frac
    elif placement == "mu_log_blend":
        mu_guard = np.exp(
            (1.0 - residual_frac) * np.log(np.maximum(mu_parent, floor))
            + residual_frac * np.log(np.maximum(mu_candidate, floor))
        )
    else:
        raise ValueError(f"Unknown placement: {placement}")
    return mu_guard.astype(float), residual_frac.astype(float)


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


def _summarize_subset(combo: dict[str, float | int | str], case_name: str, subset: str, df: pd.DataFrame, drift_d: float) -> dict[str, float | int | str]:
    worst_idx = df["abs_improvement_vs_candidate"].idxmin()
    best_idx = df["abs_improvement_vs_candidate"].idxmax()
    worst = df.loc[worst_idx]
    best = df.loc[best_idx]
    return {
        **combo,
        "case": case_name,
        "subset": subset,
        "drift_grid_D": float(drift_d),
        "count_points": int(len(df)),
        "p95_abs_delta_base": float(np.percentile(df["abs_delta_base"], 95.0)),
        "p95_abs_delta_parent": float(np.percentile(df["abs_delta_parent"], 95.0)),
        "p95_abs_delta_candidate": float(np.percentile(df["abs_delta_candidate"], 95.0)),
        "p95_abs_delta_guard": float(np.percentile(df["abs_delta_guard"], 95.0)),
        "mean_abs_improvement_vs_candidate": float(df["abs_improvement_vs_candidate"].mean()),
        "median_abs_improvement_vs_candidate": float(df["abs_improvement_vs_candidate"].median()),
        "mean_abs_improvement_vs_parent": float(df["abs_improvement_vs_parent"].mean()),
        "acceptance_flip_fraction_vs_base": float(df["acceptance_flip_fraction_vs_base"].mean()),
        "acceptance_flip_fraction_vs_candidate": float(df["acceptance_flip_fraction_vs_candidate"].mean()),
        "best_point_D": float(best["D"]),
        "best_point_eta": float(best["eta"]),
        "best_point_improvement_vs_candidate": float(best["abs_improvement_vs_candidate"]),
        "worst_point_D": float(worst["D"]),
        "worst_point_eta": float(worst["eta"]),
        "worst_point_improvement_vs_candidate": float(worst["abs_improvement_vs_candidate"]),
    }


def _aggregate_combo(case_summary: pd.DataFrame) -> dict[str, float | int | str]:
    idx_cols = ["combo_id", "placement", "peak", "center_D", "width_D"]
    ref = case_summary.iloc[0]
    d60 = case_summary[case_summary["case"].str.startswith("D60") | case_summary["case"].str.startswith("D80") | case_summary["case"].str.startswith("D100")]
    d21 = case_summary[case_summary["case"].str.startswith("D21")]

    def _subset_frame(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
        return frame.loc[frame["subset"] == subset].copy()

    d60_drift = _subset_frame(d60, "drift_strip")
    d60_reviewer = _subset_frame(d60, "reviewer_targets")
    d60_complement = _subset_frame(d60, "nonreviewer_complement")
    d21_reviewer = _subset_frame(d21, "reviewer_targets")
    d21_complement = _subset_frame(d21, "nonreviewer_complement")
    d21_all = _subset_frame(d21, "all_points")

    def _penalty(frame: pd.DataFrame) -> float:
        if frame.empty:
            return 0.0
        return float(np.max(np.maximum(frame["p95_abs_delta_guard"] - frame["p95_abs_delta_candidate"], 0.0)))

    d60_drift_gain = float(d60_drift["mean_abs_improvement_vs_candidate"].mean()) if not d60_drift.empty else 0.0
    reviewer_penalty = _penalty(d60_reviewer)
    complement_penalty = _penalty(d60_complement)
    d21_reviewer_penalty = _penalty(d21_reviewer)
    d21_complement_penalty = _penalty(d21_complement)
    d21_all_penalty = _penalty(d21_all)
    acceptance_flip_max = float(case_summary["acceptance_flip_fraction_vs_candidate"].max())
    score = (
        d60_drift_gain
        - 25.0 * reviewer_penalty
        - 10.0 * complement_penalty
        - 25.0 * d21_reviewer_penalty
        - 15.0 * d21_complement_penalty
        - 10.0 * d21_all_penalty
        - 10.0 * acceptance_flip_max
    )
    safe = (
        reviewer_penalty <= 1.0e-12
        and complement_penalty <= 1.0e-12
        and d21_reviewer_penalty <= 1.0e-12
        and d21_complement_penalty <= 1.0e-12
        and d21_all_penalty <= 1.0e-12
        and acceptance_flip_max <= 1.0e-12
    )
    return {
        **{k: ref[k] for k in idx_cols},
        "score": float(score),
        "is_safe_no_regression": float(safe),
        "d60_drift_mean_improvement_vs_candidate": d60_drift_gain,
        "d60_drift_worst_improvement_vs_candidate": float(d60_drift["worst_point_improvement_vs_candidate"].min()) if not d60_drift.empty else np.nan,
        "d60_reviewer_penalty": reviewer_penalty,
        "d60_complement_penalty": complement_penalty,
        "d21_reviewer_penalty": d21_reviewer_penalty,
        "d21_complement_penalty": d21_complement_penalty,
        "d21_all_penalty": d21_all_penalty,
        "acceptance_flip_max_vs_candidate": acceptance_flip_max,
        "d100_drift_p95_candidate": float(d60_drift.loc[d60_drift["case"] == "D100E41_monitor", "p95_abs_delta_candidate"].iloc[0]) if "D100E41_monitor" in d60_drift["case"].values else np.nan,
        "d100_drift_p95_guard": float(d60_drift.loc[d60_drift["case"] == "D100E41_monitor", "p95_abs_delta_guard"].iloc[0]) if "D100E41_monitor" in d60_drift["case"].values else np.nan,
        "d80_drift_p95_candidate": float(d60_drift.loc[d60_drift["case"] == "D80E41_monitor", "p95_abs_delta_candidate"].iloc[0]) if "D80E41_monitor" in d60_drift["case"].values else np.nan,
        "d80_drift_p95_guard": float(d60_drift.loc[d60_drift["case"] == "D80E41_monitor", "p95_abs_delta_guard"].iloc[0]) if "D80E41_monitor" in d60_drift["case"].values else np.nan,
        "d21e41_all_p95_candidate": float(d21_all.loc[d21_all["case"] == "D21E41_canonical", "p95_abs_delta_candidate"].iloc[0]) if "D21E41_canonical" in d21_all["case"].values else np.nan,
        "d21e41_all_p95_guard": float(d21_all.loc[d21_all["case"] == "D21E41_canonical", "p95_abs_delta_guard"].iloc[0]) if "D21E41_canonical" in d21_all["case"].values else np.nan,
        "d21e21_all_p95_candidate": float(d21_all.loc[d21_all["case"] == "D21E21_holdout", "p95_abs_delta_candidate"].iloc[0]) if "D21E21_holdout" in d21_all["case"].values else np.nan,
        "d21e21_all_p95_guard": float(d21_all.loc[d21_all["case"] == "D21E21_holdout", "p95_abs_delta_guard"].iloc[0]) if "D21E21_holdout" in d21_all["case"].values else np.nan,
    }


def _plot(combo_summary: pd.DataFrame, out_png: Path) -> None:
    top = combo_summary.sort_values(["is_safe_no_regression", "score", "d60_drift_mean_improvement_vs_candidate"], ascending=[False, False, False]).head(30).copy()
    if top.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), constrained_layout=True)
    sc = axes[0].scatter(
        top["d60_drift_mean_improvement_vs_candidate"],
        top["d21_all_penalty"] + top["d21_complement_penalty"] + top["d21_reviewer_penalty"],
        c=top["width_D"],
        cmap="viridis",
        s=70,
        edgecolor="black",
        linewidth=0.3,
    )
    axes[0].axhline(0.0, color="black", ls="--", alpha=0.35)
    axes[0].set_xlabel("Mean D60 drift improvement vs candidate")
    axes[0].set_ylabel("Total D21 penalty")
    axes[0].set_title("Narrow-strip tradeoff")
    axes[0].grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axes[0], label="width_D")

    ranked = top.sort_values(["is_safe_no_regression", "score"], ascending=[False, False]).head(12).copy()
    xpos = np.arange(len(ranked))
    axes[1].bar(xpos - 0.18, ranked["d100_drift_p95_candidate"] - ranked["d100_drift_p95_guard"], width=0.36, color="#4e79a7", label="D100 drift repair")
    axes[1].bar(xpos + 0.18, ranked["d80_drift_p95_candidate"] - ranked["d80_drift_p95_guard"], width=0.36, color="#e15759", label="D80 drift repair")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(ranked["combo_id"].astype(int), rotation=90, fontsize=8)
    axes[1].set_ylabel("p95 reduction vs candidate")
    axes[1].set_title("Best narrow-strip candidates")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    case_tables: dict[str, pd.DataFrame] = {}
    reviewer_ds_map: dict[str, list[float]] = {}
    drift_d_map: dict[str, float] = {}
    for case_name, cfg in CASE_CONFIG.items():
        table, reviewer_ds, drift_d = _load_case(case_name, cfg)
        case_tables[case_name] = table
        reviewer_ds_map[case_name] = reviewer_ds
        drift_d_map[case_name] = drift_d

    combo_id = 0
    case_summary_rows: list[dict[str, float | int | str]] = []
    combo_rows: list[dict[str, float | int | str]] = []
    best_detail: pd.DataFrame | None = None

    for placement in PLACEMENTS:
        for peak in PEAK_GRID:
            for center_d in CENTER_GRID:
                for width_d in WIDTH_GRID:
                    combo = {
                        "combo_id": combo_id,
                        "placement": placement,
                        "peak": float(peak),
                        "center_D": float(center_d),
                        "width_D": float(width_d),
                    }
                    per_case_rows = []
                    key_rows = []
                    for case_name, table in case_tables.items():
                        mu_guard, residual_frac = _apply_guard(
                            placement=placement,
                            mu_parent=table["mu_parent"].to_numpy(dtype=float),
                            mu_candidate=table["mu_candidate"].to_numpy(dtype=float),
                            d_vals=table["D"].to_numpy(dtype=float),
                            peak=float(peak),
                            center_d=float(center_d),
                            width_d=float(width_d),
                        )
                        detail = table.copy()
                        detail["mu_guard"] = mu_guard
                        detail["residual_frac"] = residual_frac
                        detail["chi2_guard"] = ((detail["mu_guard"] - 1.4) / 0.4) ** 2
                        detail["abs_delta_guard"] = (detail["mu_guard"] - detail["mu_full"]).abs()
                        detail["abs_improvement_vs_candidate"] = detail["abs_delta_candidate"] - detail["abs_delta_guard"]
                        detail["abs_improvement_vs_parent"] = detail["abs_delta_parent"] - detail["abs_delta_guard"]
                        detail["acceptance_flip_fraction_vs_base"] = ((detail["chi2_base"] <= 4.0) ^ (detail["chi2_guard"] <= 4.0)).astype(float)
                        detail["acceptance_flip_fraction_vs_candidate"] = ((detail["chi2_candidate"] <= 4.0) ^ (detail["chi2_guard"] <= 4.0)).astype(float)
                        subsets = ["all_points", "reviewer_targets", "nonreviewer_complement"]
                        if case_name.startswith("D60") or case_name.startswith("D80") or case_name.startswith("D100"):
                            subsets.append("drift_strip")
                        for subset in subsets:
                            sub = detail.loc[_subset_mask(detail, subset)].copy()
                            per_case_rows.append(_summarize_subset(combo, case_name, subset, sub, drift_d_map[case_name]))
                        unique_ds = np.sort(detail["D"].unique().astype(float))
                        key_ds = _snap_targets_to_grid(unique_ds, list(KEY_D_TARGETS))
                        for d_val in key_ds:
                            sub = detail.loc[np.isclose(detail["D"], d_val, atol=1.0e-9)].copy()
                            if sub.empty:
                                continue
                            key_rows.append({
                                **combo,
                                "case": case_name,
                                "D": float(d_val),
                                "slice_p95_abs_delta_parent": float(np.percentile(sub["abs_delta_parent"], 95.0)),
                                "slice_p95_abs_delta_candidate": float(np.percentile(sub["abs_delta_candidate"], 95.0)),
                                "slice_p95_abs_delta_guard": float(np.percentile(sub["abs_delta_guard"], 95.0)),
                                "slice_mean_abs_improvement_vs_candidate": float(sub["abs_improvement_vs_candidate"].mean()),
                                "slice_acceptance_flip_fraction_vs_base": float(sub["acceptance_flip_fraction_vs_base"].mean()),
                                "slice_acceptance_flip_fraction_vs_candidate": float(sub["acceptance_flip_fraction_vs_candidate"].mean()),
                            })
                    case_summary = pd.DataFrame(per_case_rows)
                    combo_rows.append(_aggregate_combo(case_summary))
                    case_summary_rows.extend(per_case_rows)
                    if best_detail is None:
                        best_detail = pd.DataFrame(key_rows)
                    combo_id += 1

    combo_summary = pd.DataFrame(combo_rows).sort_values(
        ["is_safe_no_regression", "score", "d60_drift_mean_improvement_vs_candidate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    case_summary_df = pd.DataFrame(case_summary_rows).sort_values(
        ["combo_id", "case", "subset"]
    ).reset_index(drop=True)

    best_combo_id = int(combo_summary.iloc[0]["combo_id"])
    best_case_summary = case_summary_df.loc[case_summary_df["combo_id"] == best_combo_id].copy()

    best_key_rows = []
    for case_name, table in case_tables.items():
        best_combo = combo_summary.iloc[0]
        mu_guard, residual_frac = _apply_guard(
            placement=str(best_combo["placement"]),
            mu_parent=table["mu_parent"].to_numpy(dtype=float),
            mu_candidate=table["mu_candidate"].to_numpy(dtype=float),
            d_vals=table["D"].to_numpy(dtype=float),
            peak=float(best_combo["peak"]),
            center_d=float(best_combo["center_D"]),
            width_d=float(best_combo["width_D"]),
        )
        detail = table.copy()
        detail["mu_guard"] = mu_guard
        detail["residual_frac"] = residual_frac
        detail["abs_delta_guard"] = (detail["mu_guard"] - detail["mu_full"]).abs()
        detail["abs_improvement_vs_candidate"] = detail["abs_delta_candidate"] - detail["abs_delta_guard"]
        unique_ds = np.sort(detail["D"].unique().astype(float))
        key_ds = _snap_targets_to_grid(unique_ds, list(KEY_D_TARGETS))
        for d_val in key_ds:
            sub = detail.loc[np.isclose(detail["D"], d_val, atol=1.0e-9)].copy()
            if sub.empty:
                continue
            best_key_rows.append(sub.assign(combo_id=best_combo_id))
    best_detail_df = pd.concat(best_key_rows, ignore_index=True).sort_values(["case", "D", "eta"]).reset_index(drop=True)

    summary_path = OUTDIR / f"{STEM}_summary.csv"
    case_summary_path = OUTDIR / f"{STEM}_case_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    detail_path = OUTDIR / f"{STEM}_detail.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    combo_summary.to_csv(summary_path, index=False)
    case_summary_df.to_csv(case_summary_path, index=False)
    best_case_summary.to_csv(slices_path, index=False)
    best_detail_df.to_csv(detail_path, index=False)
    _plot(combo_summary, png_path)
    meta_path.write_text(json.dumps({
        "drift_target_D": DRIFT_TARGET_D,
        "placements": list(PLACEMENTS),
        "peak_grid": list(PEAK_GRID),
        "center_grid": list(CENTER_GRID),
        "width_grid": list(WIDTH_GRID),
        "cases": {k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in cfg.items()} for k, cfg in CASE_CONFIG.items()},
        "notes": (
            "Narrow D≈7.24 strip repair audit on top of twolobe_d8compmaplate. "
            "The repair blends candidate back toward d8maplate parent only in a very narrow D-only band, "
            "with D60 micro-drift elimination prioritized and D21 regression guarded."
        ),
    }, indent=2))

    for path in [summary_path, case_summary_path, slices_path, detail_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(combo_summary.head(20).to_string(index=False))
    print(f"\nWrote summary:      {summary_path}")
    print(f"Wrote case summary: {case_summary_path}")
    print(f"Wrote slices:       {slices_path}")
    print(f"Wrote detail:       {detail_path}")


if __name__ == "__main__":
    main()
