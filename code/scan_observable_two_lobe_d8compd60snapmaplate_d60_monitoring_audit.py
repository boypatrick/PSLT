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
STEM = "observable_two_lobe_d8compd60snapmaplate_d60_monitoring_audit"

BASE_CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost"
PARENT_CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate"
CANDIDATE_CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compd60snapmaplate"

D60_TARGETS = [float(x) for x in REGIONS["D60_hotspot"]["D_targets"]]
DRIFT_TARGET_D = 7.254237288135593

CASE_CONFIG = {
    "D60E21_monitor": {
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21_refresh_20260324.csv",
        "ancestor": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D60E21_runtime_20260329_refsnapfix_d8maplate_r3.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D60E21_runtime_20260330_refsnapfix_d8comp_r1.csv",
    },
    "D60E41_monitor": {
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D60E41_20260330_crossgrid_holdout_r1.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E41_20260330_crossgrid_holdout_r1.csv",
        "ancestor": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D60E41_20260330_crossgrid_holdout_r1.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D60E41_20260330_crossgrid_holdout_r1.csv",
    },
    "D80E41_monitor": {
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D80E41_20260330_d60_micro_monitor_r1.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D80E41_20260330_d60_micro_monitor_r1.csv",
        "ancestor": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D80E41_20260330_d60_micro_monitor_r1.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D80E41_20260330_d60_micro_monitor_r1.csv",
    },
    "D100E41_monitor": {
        "full": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D100E41_20260330_d60_micro_monitor_r1.csv",
        "base": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D100E41_20260330_d60_micro_monitor_r1.csv",
        "ancestor": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D100E41_20260330_d60_micro_monitor_r1.csv",
        "parent": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate_D100E41_20260330_d60_micro_monitor_r1.csv",
    },
}


def _load_case(case_name: str, paths: dict[str, Path]) -> pd.DataFrame:
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
    drift_idx = int(np.argmin(np.abs(unique_ds - DRIFT_TARGET_D)))
    drift_d = float(unique_ds[drift_idx])
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
    merged["abs_improvement_vs_parent"] = merged["abs_delta_parent"] - merged["abs_delta_cf"]
    merged["acceptance_flip_vs_base"] = ((merged["chi2_base"] <= 4.0) ^ (merged["chi2_cf"] <= 4.0)).astype(float)
    return merged


def _snap_targets_to_grid(unique_ds: np.ndarray, targets: list[float]) -> list[float]:
    snapped: list[float] = []
    for target in targets:
        idx = int(np.argmin(np.abs(unique_ds - float(target))))
        d_val = float(unique_ds[idx])
        if not any(np.isclose(d_val, prev, atol=1.0e-9) for prev in snapped):
            snapped.append(d_val)
    return snapped


def _monitor_mask(
    df: pd.DataFrame,
    subset: str,
    reviewer_ds: list[float],
    drift_d: float,
    neighbor_ds: list[float],
) -> pd.Series:
    dvals = df["D"].to_numpy(dtype=float)
    reviewer_mask = np.zeros(len(df), dtype=bool)
    for target in reviewer_ds:
        reviewer_mask |= np.isclose(dvals, target, atol=1.0e-9)
    drift_mask = np.isclose(dvals, drift_d, atol=1.0e-9)
    neighbor_mask = np.zeros(len(df), dtype=bool)
    for d in neighbor_ds:
        neighbor_mask |= np.isclose(dvals, d, atol=1.0e-9)
    if subset == "all_points":
        return pd.Series(True, index=df.index)
    if subset == "reviewer_targets":
        return pd.Series(reviewer_mask, index=df.index)
    if subset == "nonreviewer_complement":
        return pd.Series(~reviewer_mask, index=df.index)
    if subset == "drift_strip":
        return pd.Series(drift_mask, index=df.index)
    if subset == "drift_band":
        return pd.Series(drift_mask | neighbor_mask, index=df.index)
    raise ValueError(subset)


def _summarize_subset(
    case_name: str,
    subset: str,
    df: pd.DataFrame,
    reviewer_ds: list[float],
    drift_d: float,
) -> dict[str, float | str | int]:
    if df.empty:
        return {
            "case": case_name,
            "subset": subset,
            "count_points": 0,
            "reviewer_grid_targets": "|".join(f"{x:.6f}" for x in reviewer_ds),
            "drift_target_D": float(DRIFT_TARGET_D),
            "drift_grid_D": float(drift_d),
            "p95_abs_delta_base": np.nan,
            "p95_abs_delta_parent": np.nan,
            "p95_abs_delta_cf": np.nan,
            "max_abs_delta_base": np.nan,
            "max_abs_delta_parent": np.nan,
            "max_abs_delta_cf": np.nan,
            "mean_abs_improvement_vs_parent": np.nan,
            "median_abs_improvement_vs_parent": np.nan,
            "acceptance_flip_fraction_vs_base": np.nan,
            "best_point_D": np.nan,
            "best_point_eta": np.nan,
            "best_point_improvement_vs_parent": np.nan,
            "worst_point_D": np.nan,
            "worst_point_eta": np.nan,
            "worst_point_improvement_vs_parent": np.nan,
        }
    worst_idx = df["abs_improvement_vs_parent"].idxmin()
    best_idx = df["abs_improvement_vs_parent"].idxmax()
    worst = df.loc[worst_idx]
    best = df.loc[best_idx]
    return {
        "case": case_name,
        "subset": subset,
        "count_points": int(len(df)),
        "reviewer_grid_targets": "|".join(f"{x:.6f}" for x in reviewer_ds),
        "drift_target_D": float(DRIFT_TARGET_D),
        "drift_grid_D": float(drift_d),
        "p95_abs_delta_base": float(np.percentile(df["abs_delta_base"], 95.0)),
        "p95_abs_delta_parent": float(np.percentile(df["abs_delta_parent"], 95.0)),
        "p95_abs_delta_cf": float(np.percentile(df["abs_delta_cf"], 95.0)),
        "max_abs_delta_base": float(df["abs_delta_base"].max()),
        "max_abs_delta_parent": float(df["abs_delta_parent"].max()),
        "max_abs_delta_cf": float(df["abs_delta_cf"].max()),
        "mean_abs_improvement_vs_parent": float(df["abs_improvement_vs_parent"].mean()),
        "median_abs_improvement_vs_parent": float(df["abs_improvement_vs_parent"].median()),
        "acceptance_flip_fraction_vs_base": float(df["acceptance_flip_vs_base"].mean()),
        "best_point_D": float(best["D"]),
        "best_point_eta": float(best["eta"]),
        "best_point_improvement_vs_parent": float(best["abs_improvement_vs_parent"]),
        "worst_point_D": float(worst["D"]),
        "worst_point_eta": float(worst["eta"]),
        "worst_point_improvement_vs_parent": float(worst["abs_improvement_vs_parent"]),
    }


def _slice_rows(case_name: str, df: pd.DataFrame, reviewer_ds: list[float], drift_d: float) -> pd.DataFrame:
    rows = []
    for d_val, grp in df.groupby("D", sort=True):
        d_float = float(d_val)
        rows.append({
            "case": case_name,
            "D": d_float,
            "is_reviewer_target": float(any(np.isclose(d_float, t, atol=1.0e-9) for t in reviewer_ds)),
            "is_drift_strip": float(np.isclose(d_float, drift_d, atol=1.0e-9)),
            "slice_p95_abs_delta_parent": float(np.percentile(grp["abs_delta_parent"], 95.0)),
            "slice_p95_abs_delta_cf": float(np.percentile(grp["abs_delta_cf"], 95.0)),
            "slice_mean_abs_improvement_vs_parent": float(grp["abs_improvement_vs_parent"].mean()),
            "slice_min_abs_improvement_vs_parent": float(grp["abs_improvement_vs_parent"].min()),
            "slice_max_abs_improvement_vs_parent": float(grp["abs_improvement_vs_parent"].max()),
            "slice_acceptance_flip_fraction_vs_base": float(grp["acceptance_flip_vs_base"].mean()),
        })
    return pd.DataFrame(rows)


def _eta_rows(case_name: str, drift_df: pd.DataFrame) -> pd.DataFrame:
    drift_df = drift_df.sort_values("eta")
    return pd.DataFrame({
        "case": case_name,
        "D": drift_df["D"].to_numpy(dtype=float),
        "eta": drift_df["eta"].to_numpy(dtype=float),
        "abs_delta_parent": drift_df["abs_delta_parent"].to_numpy(dtype=float),
        "abs_delta_cf": drift_df["abs_delta_cf"].to_numpy(dtype=float),
        "abs_improvement_vs_parent": drift_df["abs_improvement_vs_parent"].to_numpy(dtype=float),
    })


def _plot(slices: pd.DataFrame, eta_profiles: pd.DataFrame, out_png: Path) -> None:
    cases = list(slices["case"].drop_duplicates())
    fig, axes = plt.subplots(2, len(cases), figsize=(4.0 * len(cases), 7.2), constrained_layout=True)
    if len(cases) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for idx, case_name in enumerate(cases):
        slice_grp = slices.loc[slices["case"] == case_name].sort_values("D")
        eta_grp = eta_profiles.loc[eta_profiles["case"] == case_name].sort_values("eta")
        ax0 = axes[0, idx]
        ax0.plot(slice_grp["D"], slice_grp["slice_mean_abs_improvement_vs_parent"], marker="o", color="#59a14f")
        ax0.axhline(0.0, color="k", lw=0.8, alpha=0.4)
        for d in slice_grp.loc[slice_grp["is_reviewer_target"] > 0.5, "D"].tolist():
            ax0.axvline(d, color="#e15759", ls=":", alpha=0.12)
        for d in slice_grp.loc[slice_grp["is_drift_strip"] > 0.5, "D"].tolist():
            ax0.axvline(d, color="#4e79a7", ls="--", alpha=0.35)
        ax0.set_title(case_name)
        ax0.set_xlabel("D")
        ax0.set_ylabel("slice mean Δ(parent-candidate)")
        ax0.grid(True, alpha=0.25)

        ax1 = axes[1, idx]
        ax1.plot(eta_grp["eta"], eta_grp["abs_improvement_vs_parent"], marker="o", color="#4e79a7")
        ax1.axhline(0.0, color="k", lw=0.8, alpha=0.4)
        ax1.set_xlabel("eta")
        ax1.set_ylabel("drift-strip Δ(parent-candidate)")
        ax1.grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    path_table: dict[str, dict[str, Path]] = {}
    active_cases: dict[str, dict[str, str]] = {}
    skipped_cases: dict[str, str] = {}
    for case_name, cfg in CASE_CONFIG.items():
        missing = [key for key, path in cfg.items() if not Path(path).exists()]
        if missing:
            skipped_cases[case_name] = "missing " + ", ".join(missing)
            continue
        active_cases[case_name] = {key: str(path) for key, path in cfg.items()}
        path_table[case_name] = {key: Path(path) for key, path in cfg.items()}

    if not active_cases:
        raise RuntimeError("No monitoring cases are supported by the available full_direct localized profiles.")

    summary_rows = []
    slice_frames = []
    eta_frames = []
    detail_frames = []
    for case_name in active_cases:
        detail = _load_case(case_name, path_table[case_name])
        unique_ds = np.sort(detail["D"].unique().astype(float))
        reviewer_ds = _snap_targets_to_grid(unique_ds, D60_TARGETS)
        drift_idx = int(np.argmin(np.abs(unique_ds - DRIFT_TARGET_D)))
        drift_d = float(unique_ds[drift_idx])
        neighbor_ds = []
        if drift_idx > 0:
            neighbor_ds.append(float(unique_ds[drift_idx - 1]))
        if drift_idx + 1 < len(unique_ds):
            neighbor_ds.append(float(unique_ds[drift_idx + 1]))
        detail["drift_grid_D"] = drift_d
        detail_frames.append(detail)
        for subset in ["all_points", "reviewer_targets", "nonreviewer_complement", "drift_strip", "drift_band"]:
            sub = detail.loc[_monitor_mask(detail, subset, reviewer_ds, drift_d, neighbor_ds)].copy()
            summary_rows.append(_summarize_subset(case_name, subset, sub, reviewer_ds, drift_d))
        slice_frames.append(_slice_rows(case_name, detail, reviewer_ds, drift_d))
        eta_frames.append(_eta_rows(case_name, detail.loc[np.isclose(detail["D"], drift_d, atol=1.0e-9)].copy()))

    summary = pd.DataFrame(summary_rows)
    slices = pd.concat(slice_frames, ignore_index=True)
    eta_profiles = pd.concat(eta_frames, ignore_index=True)
    detail = pd.concat(detail_frames, ignore_index=True).sort_values(["case", "D", "eta"]).reset_index(drop=True)

    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    eta_path = OUTDIR / f"{STEM}_eta_profiles.csv"
    detail_path = OUTDIR / f"{STEM}_detail.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    eta_profiles.to_csv(eta_path, index=False)
    detail.to_csv(detail_path, index=False)
    _plot(slices, eta_profiles, png_path)
    meta_path.write_text(json.dumps({
        "base_chain_mode": BASE_CHAIN_MODE,
        "parent_chain_mode": PARENT_CHAIN_MODE,
        "candidate_chain_mode": CANDIDATE_CHAIN_MODE,
        "candidate_construction": "same localized d60 conditional blend used by observable_two_lobe_d8compd60snapmaplate_runtime_validation",
        "drift_target_D": DRIFT_TARGET_D,
        "d60_targets": D60_TARGETS,
        "active_cases": active_cases,
        "skipped_cases": skipped_cases,
        "paths": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in path_table.items()},
        "notes": "Production-style D60 monitoring for the canonical reviewer mainline twolobe_d8compd60snapmaplate, focused on whether the residual non-reviewer micro-drift near D≈7.254 stays localized after the branch switch.",
    }, indent=2))

    for path in [summary_path, slices_path, eta_path, detail_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(summary.to_string(index=False))
    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")
    print(f"Wrote eta:     {eta_path}")
    print(f"Wrote detail:  {detail_path}")


if __name__ == "__main__":
    main()
