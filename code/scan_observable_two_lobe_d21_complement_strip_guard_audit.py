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
STEM = "observable_two_lobe_d21_complement_strip_guard_audit"

D21_FULL = MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41_refresh_20260324.csv"
D21_BASE = MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_currentbase_D21E41_20260329.csv"
D21_D8MAPLATE = MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D21E41_runtime_20260329_refsnapfix_d8maplate_r3.csv"

D60_FULL = MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv"
D60_BASE = MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21_refresh_20260324.csv"
D60_D8MAPLATE = MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D60E21_runtime_20260329_refsnapfix_d8maplate_r3.csv"

PLACEMENTS = ("mu_linear_blend", "mu_log_blend")
PEAK72_GRID = (0.30, 0.50, 0.70, 0.85, 0.93, 0.97, 1.00)
WIDTH72_GRID = (0.02, 0.05, 0.10, 0.20, 0.40)
PEAK96_GRID = (0.30, 0.50, 0.70, 0.85, 0.93, 0.97, 1.00)
WIDTH96_GRID = (0.02, 0.05, 0.10, 0.20, 0.40)


def _load_d21() -> pd.DataFrame:
    full_df = pd.read_csv(D21_FULL).rename(columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"})
    base_df = pd.read_csv(D21_BASE).rename(columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"})
    cand_df = pd.read_csv(D21_D8MAPLATE).rename(columns={"mu_mumu": "mu_parent", "chi2_mumu": "chi2_parent"})
    merged = (
        full_df[["D", "eta", "mu_full", "chi2_full"]]
        .merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
        .merge(cand_df[["D", "eta", "mu_parent", "chi2_parent"]], on=["D", "eta"], how="inner")
        .sort_values(["D", "eta"])
        .reset_index(drop=True)
    )
    targets = REGIONS["D21_guard"]["D_targets"]
    merged["is_reviewer_target"] = False
    for target in targets:
        merged.loc[np.isclose(merged["D"].to_numpy(dtype=float), float(target), atol=1.0e-9), "is_reviewer_target"] = True
    return merged


def _load_d60_summary() -> dict[str, float]:
    full_df = pd.read_csv(D60_FULL).rename(columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"})
    base_df = pd.read_csv(D60_BASE).rename(columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"})
    cand_df = pd.read_csv(D60_D8MAPLATE).rename(columns={"mu_mumu": "mu_cf", "chi2_mumu": "chi2_cf"})
    merged = (
        full_df[["D", "eta", "mu_full", "chi2_full"]]
        .merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
        .merge(cand_df[["D", "eta", "mu_cf", "chi2_cf"]], on=["D", "eta"], how="inner")
        .sort_values(["D", "eta"])
        .reset_index(drop=True)
    )
    targets = REGIONS["D60_hotspot"]["D_targets"]
    mask = np.zeros(len(merged), dtype=bool)
    darr = merged["D"].to_numpy(dtype=float)
    for target in targets:
        mask |= np.isclose(darr, float(target), atol=1.0e-9)
    sub = merged.loc[mask].copy()
    abs_delta_base = np.abs(sub["mu_base"].to_numpy(dtype=float) - sub["mu_full"].to_numpy(dtype=float))
    abs_delta_cf = np.abs(sub["mu_cf"].to_numpy(dtype=float) - sub["mu_full"].to_numpy(dtype=float))
    return {
        "d60_p95_base": float(np.percentile(abs_delta_base, 95.0)),
        "d60_p95_cf": float(np.percentile(abs_delta_cf, 95.0)),
        "d60_max_base": float(np.max(abs_delta_base)),
        "d60_max_cf": float(np.max(abs_delta_cf)),
    }


def _apply_guard(
    placement: str,
    mu_base: np.ndarray,
    mu_parent: np.ndarray,
    d_vals: np.ndarray,
    peak72: float,
    width72: float,
    peak96: float,
    width96: float,
) -> tuple[np.ndarray, np.ndarray]:
    z72 = (d_vals.astype(float) - 7.2) / max(float(width72), 1.0e-12)
    z96 = (d_vals.astype(float) - 9.6) / max(float(width96), 1.0e-12)
    g72 = np.exp(-0.5 * z72 * z72)
    g96 = np.exp(-0.5 * z96 * z96)
    residual_frac = np.clip((1.0 - float(peak72) * g72) * (1.0 - float(peak96) * g96), 0.0, 1.0)
    floor = 1.0e-30
    if placement == "mu_linear_blend":
        mu_guard = mu_base + (mu_parent - mu_base) * residual_frac
    elif placement == "mu_log_blend":
        mu_guard = np.exp(
            (1.0 - residual_frac) * np.log(np.maximum(mu_base, floor))
            + residual_frac * np.log(np.maximum(mu_parent, floor))
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
    raise ValueError(subset)


def _evaluate_combo(combo: dict[str, float | int | str], d21_table: pd.DataFrame, d60_summary: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mu_guard, residual_frac = _apply_guard(
        placement=str(combo["placement"]),
        mu_base=d21_table["mu_base"].to_numpy(dtype=float),
        mu_parent=d21_table["mu_parent"].to_numpy(dtype=float),
        d_vals=d21_table["D"].to_numpy(dtype=float),
        peak72=float(combo["peak72"]),
        width72=float(combo["width72"]),
        peak96=float(combo["peak96"]),
        width96=float(combo["width96"]),
    )
    chi2_guard = ((mu_guard - 1.4) / 0.4) ** 2
    detail = d21_table.copy()
    detail["combo_id"] = int(combo["combo_id"])
    detail["placement"] = str(combo["placement"])
    detail["peak72"] = float(combo["peak72"])
    detail["width72"] = float(combo["width72"])
    detail["peak96"] = float(combo["peak96"])
    detail["width96"] = float(combo["width96"])
    detail["residual_frac"] = residual_frac
    detail["mu_guard"] = mu_guard
    detail["chi2_guard"] = chi2_guard
    detail["abs_delta_base"] = np.abs(detail["mu_base"] - detail["mu_full"])
    detail["abs_delta_parent"] = np.abs(detail["mu_parent"] - detail["mu_full"])
    detail["abs_delta_guard"] = np.abs(detail["mu_guard"] - detail["mu_full"])
    detail["abs_improvement_vs_base"] = detail["abs_delta_base"] - detail["abs_delta_guard"]
    detail["abs_improvement_vs_parent"] = detail["abs_delta_parent"] - detail["abs_delta_guard"]
    detail["improved_vs_parent"] = (detail["abs_delta_guard"] < detail["abs_delta_parent"]).astype(float)
    detail["worsened_vs_parent"] = (detail["abs_delta_guard"] > detail["abs_delta_parent"]).astype(float)
    detail["acceptance_flip_vs_base"] = ((detail["chi2_base"] <= 4.0) ^ (detail["chi2_guard"] <= 4.0)).astype(float)

    summary_rows = []
    for subset in ("all_points", "reviewer_targets", "nonreviewer_complement"):
        sub = detail.loc[_subset_mask(detail, subset)].copy()
        worst_idx = sub["abs_improvement_vs_parent"].idxmin()
        best_idx = sub["abs_improvement_vs_parent"].idxmax()
        worst = sub.loc[worst_idx]
        best = sub.loc[best_idx]
        summary_rows.append(
            {
                **combo,
                "subset": subset,
                "count_points": int(len(sub)),
                "d21_p95_abs_delta_base": float(np.percentile(sub["abs_delta_base"], 95.0)),
                "d21_p95_abs_delta_parent": float(np.percentile(sub["abs_delta_parent"], 95.0)),
                "d21_p95_abs_delta_guard": float(np.percentile(sub["abs_delta_guard"], 95.0)),
                "d21_max_abs_delta_base": float(np.max(sub["abs_delta_base"])),
                "d21_max_abs_delta_parent": float(np.max(sub["abs_delta_parent"])),
                "d21_max_abs_delta_guard": float(np.max(sub["abs_delta_guard"])),
                "d21_mean_abs_improvement_vs_parent": float(np.mean(sub["abs_improvement_vs_parent"])),
                "d21_improved_fraction_vs_parent": float(np.mean(sub["improved_vs_parent"])),
                "d21_worsened_fraction_vs_parent": float(np.mean(sub["worsened_vs_parent"])),
                "d21_acceptance_flip_fraction_vs_base": float(np.mean(sub["acceptance_flip_vs_base"])),
                "best_point_D": float(best["D"]),
                "best_point_eta": float(best["eta"]),
                "best_point_improvement_vs_parent": float(best["abs_improvement_vs_parent"]),
                "worst_point_D": float(worst["D"]),
                "worst_point_eta": float(worst["eta"]),
                "worst_point_improvement_vs_parent": float(worst["abs_improvement_vs_parent"]),
                **d60_summary,
            }
        )

    slice_rows = []
    for d_val, grp in detail.groupby("D", sort=True):
        slice_rows.append(
            {
                **combo,
                "D": float(d_val),
                "is_reviewer_target": float(grp["is_reviewer_target"].iloc[0]),
                "slice_p95_abs_delta_base": float(np.percentile(grp["abs_delta_base"], 95.0)),
                "slice_p95_abs_delta_parent": float(np.percentile(grp["abs_delta_parent"], 95.0)),
                "slice_p95_abs_delta_guard": float(np.percentile(grp["abs_delta_guard"], 95.0)),
                "slice_mean_abs_improvement_vs_parent": float(np.mean(grp["abs_improvement_vs_parent"])),
                "slice_improved_fraction_vs_parent": float(np.mean(grp["improved_vs_parent"])),
                "slice_worsened_fraction_vs_parent": float(np.mean(grp["worsened_vs_parent"])),
                "slice_acceptance_flip_fraction_vs_base": float(np.mean(grp["acceptance_flip_vs_base"])),
            }
        )

    return detail, pd.DataFrame(summary_rows), pd.DataFrame(slice_rows)


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    top = summary.copy().head(24)
    if top.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.2), constrained_layout=True)
    x = np.arange(len(top))
    axes[0].bar(x - 0.18, top["d21_p95_abs_delta_guard"], width=0.36, label="D21 complement", color="#4e79a7")
    axes[0].bar(x + 0.18, top["d60_p95_cf"], width=0.36, label="D60 reviewer", color="#e15759")
    axes[0].axhline(float(top["d21_p95_abs_delta_parent"].iloc[0]), ls="--", color="#4e79a7", alpha=0.5)
    axes[0].axhline(float(top["d60_p95_cf"].iloc[0]), ls="--", color="#e15759", alpha=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(top["combo_id"].astype(int), rotation=90, fontsize=8)
    axes[0].set_title("Complement repair vs D60 preservation")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(
        top["d21_p95_abs_delta_guard"],
        top["d21_acceptance_flip_fraction_vs_base"],
        c=top["peak72"],
        cmap="viridis",
        s=45,
    )
    axes[1].axvline(float(top["d21_p95_abs_delta_parent"].iloc[0]), ls="--", color="#4e79a7", alpha=0.5)
    axes[1].set_xlabel("D21 complement p95 |Δμ|")
    axes[1].set_ylabel("Acceptance flip fraction")
    axes[1].set_title("Complement tradeoff")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    d21_table = _load_d21()
    d60_summary = _load_d60_summary()

    combos = []
    combo_id = 1
    for placement in PLACEMENTS:
        for peak72 in PEAK72_GRID:
            for width72 in WIDTH72_GRID:
                for peak96 in PEAK96_GRID:
                    for width96 in WIDTH96_GRID:
                        combos.append(
                            {
                                "combo_id": combo_id,
                                "placement": placement,
                                "peak72": float(peak72),
                                "width72": float(width72),
                                "peak96": float(peak96),
                                "width96": float(width96),
                            }
                        )
                        combo_id += 1

    detail_frames = []
    summary_frames = []
    slice_frames = []
    for combo in combos:
        detail_df, summary_df, slices_df = _evaluate_combo(combo, d21_table, d60_summary)
        detail_frames.append(detail_df)
        summary_frames.append(summary_df)
        slice_frames.append(slices_df)

    detail = pd.concat(detail_frames, ignore_index=True)
    summary_long = pd.concat(summary_frames, ignore_index=True)
    slices = pd.concat(slice_frames, ignore_index=True)

    summary = summary_long.copy()
    comp = summary[summary["subset"] == "nonreviewer_complement"].copy().set_index("combo_id")
    targets = summary[summary["subset"] == "reviewer_targets"].copy().set_index("combo_id")
    allp = summary[summary["subset"] == "all_points"].copy().set_index("combo_id")
    joined = comp[
        [
            "placement",
            "peak72",
            "width72",
            "peak96",
            "width96",
            "d21_p95_abs_delta_base",
            "d21_p95_abs_delta_parent",
            "d21_p95_abs_delta_guard",
            "d21_max_abs_delta_parent",
            "d21_max_abs_delta_guard",
            "d21_mean_abs_improvement_vs_parent",
            "d21_improved_fraction_vs_parent",
            "d21_worsened_fraction_vs_parent",
            "d21_acceptance_flip_fraction_vs_base",
            "d60_p95_base",
            "d60_p95_cf",
            "worst_point_D",
            "worst_point_eta",
            "worst_point_improvement_vs_parent",
        ]
    ].join(
        targets[
            [
                "d21_p95_abs_delta_parent",
                "d21_p95_abs_delta_guard",
                "d21_acceptance_flip_fraction_vs_base",
            ]
        ].rename(
            columns={
                "d21_p95_abs_delta_parent": "targets_p95_parent",
                "d21_p95_abs_delta_guard": "targets_p95_guard",
                "d21_acceptance_flip_fraction_vs_base": "targets_flip_guard",
            }
        )
    ).join(
        allp[
            [
                "d21_p95_abs_delta_parent",
                "d21_p95_abs_delta_guard",
                "d21_acceptance_flip_fraction_vs_base",
            ]
        ].rename(
            columns={
                "d21_p95_abs_delta_parent": "all_p95_parent",
                "d21_p95_abs_delta_guard": "all_p95_guard",
                "d21_acceptance_flip_fraction_vs_base": "all_flip_guard",
            }
        )
    )
    joined["preserves_d60"] = np.isclose(joined["d60_p95_cf"], joined["d60_p95_cf"].iloc[0], atol=1.0e-12)
    joined["beats_parent_complement"] = joined["d21_p95_abs_delta_guard"] < joined["d21_p95_abs_delta_parent"]
    joined["does_not_worsen_targets"] = joined["targets_p95_guard"] <= joined["targets_p95_parent"] + 1.0e-12
    joined["better_allpoints"] = joined["all_p95_guard"] < joined["all_p95_parent"]
    joined["joint_objective"] = (
        joined["d21_p95_abs_delta_guard"]
        + 10.0 * joined["d21_acceptance_flip_fraction_vs_base"]
        + 5.0 * np.maximum(joined["targets_p95_guard"] - joined["targets_p95_parent"], 0.0)
    )
    joined = joined.sort_values(
        ["joint_objective", "d21_p95_abs_delta_guard", "targets_p95_guard", "peak72", "peak96"]
    ).reset_index()

    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    detail_path = OUTDIR / f"{STEM}_detail.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    joined.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    detail.to_csv(detail_path, index=False)
    _plot(joined, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "parent_candidate": str(D21_D8MAPLATE),
                "base_map": str(D21_BASE),
                "full_map": str(D21_FULL),
                "d60_reference": str(D60_D8MAPLATE),
                "notes": (
                    "Late map-layer two-strip repair on top of twolobe_d8maplate. "
                    "Only D21 complement strips around D=7.2 and D=9.6 are retouched; D60 is kept equal to d8maplate."
                ),
            },
            indent=2,
        )
    )

    for path in [summary_path, slices_path, detail_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(joined.head(12).to_string(index=False))
    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")
    print(f"Wrote detail:  {detail_path}")


if __name__ == "__main__":
    main()
