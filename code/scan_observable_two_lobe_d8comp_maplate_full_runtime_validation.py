#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
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
STEM = "observable_two_lobe_d8comp_maplate_full_runtime_validation"

CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate"
TAG_SUFFIX = "20260330_refsnapfix_d8comp_r1"

CASE_CONFIG = {
    "D60E21_fullmap": {
        "region_key": "D60_hotspot",
        "tag": f"chain_mode_{CHAIN_MODE}_D60E21_runtime_{TAG_SUFFIX}",
        "d_num": 60,
        "eta_num": 21,
        "full_map": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv",
        "base_map": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21_refresh_20260324.csv",
        "parent_map": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D60E21_runtime_20260329_refsnapfix_d8maplate_r3.csv",
    },
    "D21E41_fullmap": {
        "region_key": "D21_guard",
        "tag": f"chain_mode_{CHAIN_MODE}_D21E41_runtime_{TAG_SUFFIX}",
        "d_num": 21,
        "eta_num": 41,
        "full_map": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41_refresh_20260324.csv",
        "base_map": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_currentbase_D21E41_20260329.csv",
        "parent_map": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate_D21E41_runtime_20260329_refsnapfix_d8maplate_r3.csv",
    },
}


def _run_map(chain_mode: str, tag: str, d_num: int, eta_num: int) -> Path:
    out_path = MAP_DIR / f"hll_signal_strength_map_{tag}.csv"
    if out_path.exists():
        return out_path
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(Path("/tmp") / "pslt_mplconfig")
    env["XDG_CACHE_HOME"] = str(Path("/tmp") / "pslt_cache")
    cmd = [
        sys.executable,
        str(CODE / "scan_hll_signal_strengths.py"),
        "--chain-mode",
        str(chain_mode),
        "--d-min",
        "4.0",
        "--d-max",
        "20.0",
        "--d-num",
        str(int(d_num)),
        "--eta-min",
        "0.2",
        "--eta-max",
        "4.0",
        "--eta-num",
        str(int(eta_num)),
        "--tag",
        tag,
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return out_path


def _load_case(cfg: dict[str, object], candidate_map: Path) -> pd.DataFrame:
    full_df = pd.read_csv(cfg["full_map"]).rename(columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"})
    base_df = pd.read_csv(cfg["base_map"]).rename(columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"})
    parent_df = pd.read_csv(cfg["parent_map"]).rename(columns={"mu_mumu": "mu_parent", "chi2_mumu": "chi2_parent"})
    cand_df = pd.read_csv(candidate_map).rename(columns={"mu_mumu": "mu_cf", "chi2_mumu": "chi2_cf"})
    merged = (
        full_df[["D", "eta", "mu_full", "chi2_full"]]
        .merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
        .merge(parent_df[["D", "eta", "mu_parent", "chi2_parent"]], on=["D", "eta"], how="inner")
        .merge(cand_df[["D", "eta", "mu_cf", "chi2_cf"]], on=["D", "eta"], how="inner")
        .sort_values(["D", "eta"])
        .reset_index(drop=True)
    )
    merged["abs_delta_base"] = (merged["mu_base"] - merged["mu_full"]).abs()
    merged["abs_delta_parent"] = (merged["mu_parent"] - merged["mu_full"]).abs()
    merged["abs_delta_cf"] = (merged["mu_cf"] - merged["mu_full"]).abs()
    merged["abs_improvement_vs_base"] = merged["abs_delta_base"] - merged["abs_delta_cf"]
    merged["abs_improvement_vs_parent"] = merged["abs_delta_parent"] - merged["abs_delta_cf"]
    merged["improved_vs_base"] = (merged["abs_delta_cf"] < merged["abs_delta_base"]).astype(float)
    merged["improved_vs_parent"] = (merged["abs_delta_cf"] < merged["abs_delta_parent"]).astype(float)
    merged["worsened_vs_parent"] = (merged["abs_delta_cf"] > merged["abs_delta_parent"]).astype(float)
    merged["acceptance_flip_vs_base"] = ((merged["chi2_base"] <= 4.0) ^ (merged["chi2_cf"] <= 4.0)).astype(float)
    return merged


def _subset_mask(df: pd.DataFrame, targets: list[float], subset: str) -> pd.Series:
    dvals = df["D"].to_numpy(dtype=float)
    target_mask = np.zeros(len(df), dtype=bool)
    for target in targets:
        target_mask |= np.isclose(dvals, float(target), atol=1.0e-9)
    if subset == "all_points":
        return pd.Series(True, index=df.index)
    if subset == "reviewer_targets":
        return pd.Series(target_mask, index=df.index)
    if subset == "nonreviewer_complement":
        return pd.Series(~target_mask, index=df.index)
    raise ValueError(subset)


def _summarize_subset(case_name: str, subset: str, df: pd.DataFrame) -> dict[str, float | str | int]:
    worst_idx = df["abs_improvement_vs_parent"].idxmin()
    best_idx = df["abs_improvement_vs_parent"].idxmax()
    worst = df.loc[worst_idx]
    best = df.loc[best_idx]
    return {
        "case": case_name,
        "subset": subset,
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


def _slice_rows(case_name: str, targets: list[float], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_set = {float(x) for x in targets}
    for d_val, grp in df.groupby("D", sort=True):
        d_float = float(d_val)
        rows.append(
            {
                "case": case_name,
                "D": d_float,
                "is_reviewer_target": float(any(np.isclose(d_float, t, atol=1.0e-9) for t in target_set)),
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
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    axes = axes.flatten()
    palette = {"base": "#4e79a7", "parent": "#e15759", "candidate": "#59a14f"}
    for ax, (case_name, grp) in zip(axes, slices.groupby("case", sort=False)):
        grp = grp.sort_values("D")
        ax.plot(grp["D"], grp["slice_p95_abs_delta_base"], marker="o", color=palette["base"], label="base")
        ax.plot(grp["D"], grp["slice_p95_abs_delta_parent"], marker="o", color=palette["parent"], label="d8maplate parent")
        ax.plot(grp["D"], grp["slice_p95_abs_delta_cf"], marker="o", color=palette["candidate"], label="d8compmaplate")
        for d in grp.loc[grp["is_reviewer_target"] > 0.5, "D"].tolist():
            ax.axvline(d, color="k", ls=":", alpha=0.15)
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

    candidate_maps = {
        case_name: _run_map(
            CHAIN_MODE,
            str(cfg["tag"]),
            int(cfg["d_num"]),
            int(cfg["eta_num"]),
        )
        for case_name, cfg in CASE_CONFIG.items()
    }

    summary_rows = []
    slice_frames = []
    detail_frames = []
    for case_name, cfg in CASE_CONFIG.items():
        region_key = str(cfg["region_key"])
        targets = REGIONS[region_key]["D_targets"]
        detail = _load_case(cfg, candidate_maps[case_name])
        detail["case"] = case_name
        detail["is_reviewer_target"] = False
        for target in targets:
            detail.loc[np.isclose(detail["D"].to_numpy(dtype=float), float(target), atol=1.0e-9), "is_reviewer_target"] = True
        detail_frames.append(detail)
        for subset in ["all_points", "reviewer_targets", "nonreviewer_complement"]:
            sub = detail.loc[_subset_mask(detail, targets, subset)].copy()
            summary_rows.append(_summarize_subset(case_name, subset, sub))
        slice_frames.append(_slice_rows(case_name, targets, detail))

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
                "chain_mode": CHAIN_MODE,
                "candidate_maps": {k: str(v) for k, v in candidate_maps.items()},
                "cases": {k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in cfg.items()} for k, cfg in CASE_CONFIG.items()},
                "reviewer_regions": {k: {"D_targets": [float(x) for x in v["D_targets"]]} for k, v in REGIONS.items()},
                "notes": (
                    "Formal full-map validation for twolobe_d8compmaplate against current mainline base and the "
                    "twolobe_d8maplate parent. The candidate keeps D60 fixed to the repaired parent while applying "
                    "late map-layer complement-strip blends at D=7.2 and D=9.6."
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
