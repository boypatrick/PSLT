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

from scan_prolate_hybrid_reviewer_counterfactual_audit import REGIONS, _subset_rows

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
MAP_DIR = ROOT / "output" / "hll_signal_strength"
STEM = "observable_two_lobe_maplate_currentbase_runtime_validation"

CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_maplate"
BASE_CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost"
TAG_SUFFIX = "20260329_refsnapfix"

CASE_CONFIG = {
    "D60_hotspot": {
        "tag": f"chain_mode_{CHAIN_MODE}_D60E21_runtime_{TAG_SUFFIX}",
        "d_num": 60,
        "eta_num": 21,
        "full_map": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv",
        "base_map": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21_refresh_20260324.csv",
        "runtime_map": MAP_DIR / f"hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_D60E21_runtime_{TAG_SUFFIX}.csv",
    },
    "D21_guard": {
        "tag": f"chain_mode_{CHAIN_MODE}_D21E41_runtime_{TAG_SUFFIX}",
        "base_tag": f"chain_mode_{BASE_CHAIN_MODE}_currentbase_D21E41_20260329",
        "d_num": 21,
        "eta_num": 41,
        "full_map": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41_refresh_20260324.csv",
        "runtime_map": MAP_DIR / f"hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_D21E41_runtime_{TAG_SUFFIX}.csv",
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


def _evaluate_region(region: str, cfg: dict[str, object], candidate_map: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    region_cfg = REGIONS[region]
    full_df = _subset_rows(Path(cfg["full_map"]), region_cfg["D_targets"]).rename(
        columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"}
    )
    base_df = _subset_rows(Path(cfg["base_map"]), region_cfg["D_targets"]).rename(
        columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"}
    )
    runtime_df = _subset_rows(Path(cfg["runtime_map"]), region_cfg["D_targets"]).rename(
        columns={"mu_mumu": "mu_runtime", "chi2_mumu": "chi2_runtime"}
    )
    cand_df = _subset_rows(candidate_map, region_cfg["D_targets"]).rename(
        columns={"mu_mumu": "mu_cf", "chi2_mumu": "chi2_cf"}
    )
    merged = (
        full_df.merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
        .merge(runtime_df[["D", "eta", "mu_runtime", "chi2_runtime"]], on=["D", "eta"], how="inner")
        .merge(cand_df[["D", "eta", "mu_cf", "chi2_cf"]], on=["D", "eta"], how="inner")
    )
    merged["abs_delta_base"] = (merged["mu_base"] - merged["mu_full"]).abs()
    merged["abs_delta_runtime"] = (merged["mu_runtime"] - merged["mu_full"]).abs()
    merged["abs_delta_cf"] = (merged["mu_cf"] - merged["mu_full"]).abs()
    merged["abs_improvement_vs_base"] = merged["abs_delta_base"] - merged["abs_delta_cf"]
    merged["abs_improvement_vs_runtime"] = merged["abs_delta_runtime"] - merged["abs_delta_cf"]
    merged["cf_shift_vs_runtime"] = merged["mu_cf"] - merged["mu_runtime"]
    merged["improved_vs_base"] = (merged["abs_delta_cf"] < merged["abs_delta_base"]).astype(float)
    merged["improved_vs_runtime"] = (merged["abs_delta_cf"] < merged["abs_delta_runtime"]).astype(float)
    merged["acceptance_flip_vs_base"] = ((merged["chi2_base"] <= 4.0) ^ (merged["chi2_cf"] <= 4.0)).astype(float)

    region_summary = pd.DataFrame(
        [
            {
                "region": region,
                "candidate_map": str(candidate_map),
                "full_map": str(cfg["full_map"]),
                "base_map": str(cfg["base_map"]),
                "runtime_map": str(cfg["runtime_map"]),
                "p95_abs_delta_base": float(np.percentile(merged["abs_delta_base"], 95.0)),
                "p95_abs_delta_runtime": float(np.percentile(merged["abs_delta_runtime"], 95.0)),
                "p95_abs_delta_cf": float(np.percentile(merged["abs_delta_cf"], 95.0)),
                "max_abs_delta_base": float(np.max(merged["abs_delta_base"])),
                "max_abs_delta_runtime": float(np.max(merged["abs_delta_runtime"])),
                "max_abs_delta_cf": float(np.max(merged["abs_delta_cf"])),
                "mean_abs_improvement_vs_base": float(np.mean(merged["abs_improvement_vs_base"])),
                "mean_abs_improvement_vs_runtime": float(np.mean(merged["abs_improvement_vs_runtime"])),
                "improved_fraction_vs_base": float(np.mean(merged["improved_vs_base"])),
                "improved_fraction_vs_runtime": float(np.mean(merged["improved_vs_runtime"])),
                "acceptance_flip_fraction": float(np.mean(merged["acceptance_flip_vs_base"])),
                "best_point_D": float(merged.loc[merged["abs_improvement_vs_base"].idxmax(), "D"]),
                "best_point_eta": float(merged.loc[merged["abs_improvement_vs_base"].idxmax(), "eta"]),
                "best_point_improvement_vs_base": float(np.max(merged["abs_improvement_vs_base"])),
                "worst_point_D": float(merged.loc[merged["abs_improvement_vs_base"].idxmin(), "D"]),
                "worst_point_eta": float(merged.loc[merged["abs_improvement_vs_base"].idxmin(), "eta"]),
                "worst_point_improvement_vs_base": float(np.min(merged["abs_improvement_vs_base"])),
            }
        ]
    )

    slices = []
    for d_val, grp in merged.groupby("D", sort=True):
        slices.append(
            {
                "region": region,
                "D": float(d_val),
                "slice_p95_abs_delta_base": float(np.percentile(grp["abs_delta_base"], 95.0)),
                "slice_p95_abs_delta_runtime": float(np.percentile(grp["abs_delta_runtime"], 95.0)),
                "slice_p95_abs_delta_cf": float(np.percentile(grp["abs_delta_cf"], 95.0)),
                "slice_mean_abs_improvement_vs_base": float(np.mean(grp["abs_improvement_vs_base"])),
                "slice_mean_abs_improvement_vs_runtime": float(np.mean(grp["abs_improvement_vs_runtime"])),
                "slice_improved_fraction_vs_base": float(np.mean(grp["improved_vs_base"])),
                "slice_improved_fraction_vs_runtime": float(np.mean(grp["improved_vs_runtime"])),
            }
        )
    return region_summary, pd.DataFrame(slices), merged


def _plot(summary: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    regions = summary["region"].tolist()
    x = np.arange(len(regions))
    width = 0.26
    axes[0].bar(x - width, summary["p95_abs_delta_base"], width=width, label="base")
    axes[0].bar(x, summary["p95_abs_delta_runtime"], width=width, label="two-lobe runtime")
    axes[0].bar(x + width, summary["p95_abs_delta_cf"], width=width, label="map-layer runtime")
    axes[0].set_xticks(x, regions)
    axes[0].set_title("Reviewer p95 |Δμ|")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2.0, summary["improved_fraction_vs_base"], width=width, label="vs base")
    axes[1].bar(x + width / 2.0, summary["improved_fraction_vs_runtime"], width=width, label="vs runtime")
    axes[1].set_xticks(x, regions)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Improved fraction")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].bar(x, summary["acceptance_flip_fraction"], width=0.45, color="#e15759")
    axes[2].set_xticks(x, regions)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Acceptance flips vs base")
    axes[2].grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    base_maps = {
        region: (
            _run_map(BASE_CHAIN_MODE, str(cfg["base_tag"]), int(cfg["d_num"]), int(cfg["eta_num"]))
            if "base_tag" in cfg
            else Path(cfg["base_map"])
        )
        for region, cfg in CASE_CONFIG.items()
    }
    candidate_maps = {
        region: _run_map(CHAIN_MODE, str(cfg["tag"]), int(cfg["d_num"]), int(cfg["eta_num"]))
        for region, cfg in CASE_CONFIG.items()
    }

    summary_rows = []
    slice_rows = []
    detail_rows = []
    for region, cfg in CASE_CONFIG.items():
        cfg_eval = dict(cfg)
        cfg_eval["base_map"] = base_maps[region]
        region_summary, region_slices, detail = _evaluate_region(region, cfg_eval, candidate_maps[region])
        summary_rows.append(region_summary)
        slice_rows.append(region_slices)
        detail_rows.append(detail)

    summary = pd.concat(summary_rows, ignore_index=True)
    slices = pd.concat(slice_rows, ignore_index=True)
    detail = pd.concat(detail_rows, ignore_index=True)

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
                "chain_mode": CHAIN_MODE,
                "candidate_maps": {k: str(v) for k, v in candidate_maps.items()},
                "base_maps": {k: str(v) for k, v in base_maps.items()},
                "baseline_maps": {
                    k: {"full": str(v["full_map"]), "runtime": str(v["runtime_map"])}
                    for k, v in CASE_CONFIG.items()
                },
                "notes": (
                    "Validation of the map-layer experimental runtime two-lobe mode against current freshly generated mainline base maps and the raw two-lobe runtime mode. "
                    "This removes dependence on stale static D21 baselines so the late map-layer correction is compared apples-to-apples against the current code path."
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
