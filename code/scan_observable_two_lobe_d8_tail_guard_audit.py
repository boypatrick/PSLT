#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
MAP_DIR = ROOT / "output" / "hll_signal_strength"
STEM = "observable_two_lobe_d8_tail_guard_audit"

CASE_CONFIG = {
    "D60_hotspot": {
        "full_map": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv",
        "base_map": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21_refresh_20260324.csv",
        "runtime_map": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_D60E21_runtime_20260329_refsnapfix.csv",
        "D_targets": (4.813559322033898, 5.627118644067797, 6.169491525423729, 6.440677966101695, 6.711864406779661),
    },
    "D21_guard": {
        "full_map": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41_refresh_20260324.csv",
        "base_map": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_currentbase_D21E41_20260329.csv",
        "runtime_map": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_D21E41_runtime_20260329_refsnapfix.csv",
        "D_targets": (4.0, 6.4, 8.0),
    },
}

PLACEMENTS = ("mu_linear_blend", "mu_log_blend", "mu_exp_notch")
PEAK_GRID = (0.50, 0.70, 0.85, 0.93, 0.97, 0.99, 1.00)
CENTER_GRID = (7.95, 8.00, 8.05)
WIDTH_GRID = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40)


def _subset_rows(csv_path: Path, d_targets: tuple[float, ...]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    d_arr = df["D"].to_numpy(dtype=float)
    keep_idx = []
    for target in d_targets:
        keep_idx.append(int(np.argmin(np.abs(d_arr - float(target)))))
    keep_d = sorted({float(d_arr[idx]) for idx in keep_idx})
    return df[np.isclose(df["D"].to_numpy(dtype=float)[:, None], np.array(keep_d)[None, :], atol=1.0e-9).any(axis=1)].copy()


def _region_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for region, cfg in CASE_CONFIG.items():
        full_df = _subset_rows(Path(cfg["full_map"]), tuple(cfg["D_targets"])).rename(
            columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"}
        )
        base_df = _subset_rows(Path(cfg["base_map"]), tuple(cfg["D_targets"])).rename(
            columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"}
        )
        runtime_df = _subset_rows(Path(cfg["runtime_map"]), tuple(cfg["D_targets"])).rename(
            columns={"mu_mumu": "mu_runtime", "chi2_mumu": "chi2_runtime"}
        )
        tables[region] = (
            full_df.merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
            .merge(runtime_df[["D", "eta", "mu_runtime", "chi2_runtime"]], on=["D", "eta"], how="inner")
            .sort_values(["D", "eta"])
            .reset_index(drop=True)
        )
    return tables


def _apply_guard(placement: str, mu_base: np.ndarray, mu_runtime: np.ndarray, d_vals: np.ndarray, peak: float, center_d: float, width_d: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = (d_vals.astype(float) - float(center_d)) / max(float(width_d), 1.0e-12)
    gaussian = np.exp(-0.5 * z * z)
    residual_frac = np.clip(1.0 - float(peak) * gaussian, 0.0, 1.0)
    floor = 1.0e-30
    if placement == "mu_linear_blend":
        mu_guard = mu_base + (mu_runtime - mu_base) * residual_frac
    elif placement == "mu_log_blend":
        mu_guard = np.exp(
            (1.0 - residual_frac) * np.log(np.maximum(mu_base, floor))
            + residual_frac * np.log(np.maximum(mu_runtime, floor))
        )
    elif placement == "mu_exp_notch":
        mu_guard = mu_runtime * np.exp(-float(peak) * gaussian)
    else:
        raise ValueError(f"Unknown placement: {placement}")
    return mu_guard.astype(float), gaussian.astype(float), residual_frac.astype(float)


def _evaluate_combo(combo: dict[str, float | int | str], region_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []
    slice_rows = []
    for region, merged in region_tables.items():
        mu_guard, gaussian, residual_frac = _apply_guard(
            placement=str(combo["placement"]),
            mu_base=merged["mu_base"].to_numpy(dtype=float),
            mu_runtime=merged["mu_runtime"].to_numpy(dtype=float),
            d_vals=merged["D"].to_numpy(dtype=float),
            peak=float(combo["peak"]),
            center_d=float(combo["center_D"]),
            width_d=float(combo["width_D"]),
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
                "improved_vs_base": (abs_delta_guard < abs_delta_base).astype(float),
                "improved_vs_runtime": (abs_delta_guard < abs_delta_runtime).astype(float),
                "acceptance_flip_vs_base": ((merged["chi2_base"].to_numpy(dtype=float) <= 4.0) ^ (chi2_guard <= 4.0)).astype(float),
                "acceptance_flip_vs_runtime": ((merged["chi2_runtime"].to_numpy(dtype=float) <= 4.0) ^ (chi2_guard <= 4.0)).astype(float),
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
                "mean_abs_improvement_vs_base": float(np.mean(abs_improvement_vs_base)),
                "mean_abs_improvement_vs_runtime": float(np.mean(abs_improvement_vs_runtime)),
                "improved_fraction_vs_base": float(np.mean(region_detail["improved_vs_base"])),
                "improved_fraction_vs_runtime": float(np.mean(region_detail["improved_vs_runtime"])),
                "acceptance_flip_fraction_vs_base": float(np.mean(region_detail["acceptance_flip_vs_base"])),
                "acceptance_flip_fraction_vs_runtime": float(np.mean(region_detail["acceptance_flip_vs_runtime"])),
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
                    "slice_improved_fraction_vs_base": float(np.mean(grp["improved_vs_base"])),
                    "slice_improved_fraction_vs_runtime": float(np.mean(grp["improved_vs_runtime"])),
                }
            )

    return pd.concat(detail_rows, ignore_index=True), pd.DataFrame(summary_rows), pd.DataFrame(slice_rows)


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    top = summary.head(24)
    if top.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.2), constrained_layout=True)
    x = np.arange(len(top))
    axes[0].bar(x - 0.18, top["p95_abs_delta_guard_D60_hotspot"], width=0.36, label="D60", color="#e15759")
    axes[0].bar(x + 0.18, top["p95_abs_delta_guard_D21_guard"], width=0.36, label="D21", color="#4e79a7")
    axes[0].axhline(float(top["p95_abs_delta_base_D60_hotspot"].iloc[0]), ls="--", color="#e15759", alpha=0.5)
    axes[0].axhline(float(top["p95_abs_delta_base_D21_guard"].iloc[0]), ls="--", color="#4e79a7", alpha=0.5)
    axes[0].set_title("D=8 strip guard p95 |Δμ|")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(top["combo_id"].astype(int), rotation=90, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(
        top["p95_abs_delta_guard_D60_hotspot"],
        top["p95_abs_delta_guard_D21_guard"],
        c=top["peak"],
        cmap="viridis",
        s=45,
    )
    axes[1].axvline(float(top["p95_abs_delta_base_D60_hotspot"].iloc[0]), ls="--", color="#e15759", alpha=0.5)
    axes[1].axhline(float(top["p95_abs_delta_base_D21_guard"].iloc[0]), ls="--", color="#4e79a7", alpha=0.5)
    axes[1].set_xlabel("D60 p95 |Δμ|")
    axes[1].set_ylabel("D21 p95 |Δμ|")
    axes[1].set_title("Tradeoff vs mainline")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    region_tables = _region_tables()
    combos = []
    combo_id = 1
    for placement in PLACEMENTS:
        for peak in PEAK_GRID:
            for center_d in CENTER_GRID:
                for width_d in WIDTH_GRID:
                    combos.append(
                        {
                            "combo_id": combo_id,
                            "placement": placement,
                            "peak": float(peak),
                            "center_D": float(center_d),
                            "width_D": float(width_d),
                        }
                    )
                    combo_id += 1

    detail_frames = []
    summary_frames = []
    slice_frames = []
    for combo in combos:
        detail_df, summary_df, slices_df = _evaluate_combo(combo, region_tables)
        detail_frames.append(detail_df)
        summary_frames.append(summary_df)
        slice_frames.append(slices_df)

    detail = pd.concat(detail_frames, ignore_index=True)
    summary_long = pd.concat(summary_frames, ignore_index=True)
    slices = pd.concat(slice_frames, ignore_index=True)

    summary = (
        summary_long.pivot_table(
            index=["combo_id", "placement", "peak", "center_D", "width_D"],
            columns="region",
            values=[
                "p95_abs_delta_base",
                "p95_abs_delta_runtime",
                "p95_abs_delta_guard",
                "mean_abs_improvement_vs_base",
                "mean_abs_improvement_vs_runtime",
                "improved_fraction_vs_base",
                "improved_fraction_vs_runtime",
                "acceptance_flip_fraction_vs_base",
                "acceptance_flip_fraction_vs_runtime",
            ],
        )
        .sort_index(axis=1)
    )
    summary.columns = [f"{metric}_{region}" for metric, region in summary.columns]
    summary = summary.reset_index()
    summary["both_better_than_base"] = (
        (summary["p95_abs_delta_guard_D60_hotspot"] < summary["p95_abs_delta_base_D60_hotspot"])
        & (summary["p95_abs_delta_guard_D21_guard"] < summary["p95_abs_delta_base_D21_guard"])
    )
    summary["both_better_than_runtime"] = (
        (summary["p95_abs_delta_guard_D60_hotspot"] < summary["p95_abs_delta_runtime_D60_hotspot"])
        & (summary["p95_abs_delta_guard_D21_guard"] < summary["p95_abs_delta_runtime_D21_guard"])
    )
    summary = summary.sort_values(
        [
            "p95_abs_delta_guard_D21_guard",
            "acceptance_flip_fraction_vs_base_D21_guard",
            "p95_abs_delta_guard_D60_hotspot",
            "combo_id",
        ],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    detail_path = OUTDIR / f"{STEM}_detail.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    detail.to_csv(detail_path, index=False)
    _plot(summary, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "runtime_maps": {k: str(v["runtime_map"]) for k, v in CASE_CONFIG.items()},
                "base_maps": {k: str(v["base_map"]) for k, v in CASE_CONFIG.items()},
                "full_maps": {k: str(v["full_map"]) for k, v in CASE_CONFIG.items()},
                "placements": list(PLACEMENTS),
                "peak_grid": list(PEAK_GRID),
                "center_grid": list(CENTER_GRID),
                "width_grid": list(WIDTH_GRID),
                "notes": "Narrow late-guard shortlist targeting the corrected raw twolobe D=8.0 tail and D21 acceptance flips.",
            },
            indent=2,
        )
    )

    for path in [summary_path, slices_path, detail_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(summary.head(20).to_string(index=False))
    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")
    print(f"Wrote detail:  {detail_path}")


if __name__ == "__main__":
    main()
