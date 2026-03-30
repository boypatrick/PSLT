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
STEM = "observable_two_lobe_d8comp_maplate_d60_monitoring_audit"

BASE_CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost"
PARENT_CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8maplate"
CANDIDATE_CHAIN_MODE = "cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_twolobe_d8compmaplate"
TAG_SUFFIX = "20260330_d60_micro_monitor_r1"

D60_TARGETS = [float(x) for x in REGIONS["D60_hotspot"]["D_targets"]]
DRIFT_TARGET_D = 7.254237288135593

CASE_CONFIG = {
    "D60E21_monitor": {"d_num": 60, "eta_num": 21},
    "D60E41_monitor": {"d_num": 60, "eta_num": 41},
    "D80E41_monitor": {"d_num": 80, "eta_num": 41},
    "D100E41_monitor": {"d_num": 100, "eta_num": 41},
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
        chain_mode,
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


def _full_direct_supported(d_num: int) -> bool:
    chi = ROOT / "output" / "kinetic_action_chain" / f"localized_direct_profiles_chi_Dgrid{int(d_num)}.csv"
    gamma = ROOT / "output" / "kinetic_action_chain" / f"localized_direct_profiles_superrad_Dgrid{int(d_num)}.csv"
    return chi.exists() and gamma.exists()


def _load_case(case_name: str, paths: dict[str, Path]) -> pd.DataFrame:
    full_df = pd.read_csv(paths["full"]).rename(columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"})
    base_df = pd.read_csv(paths["base"]).rename(columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"})
    parent_df = pd.read_csv(paths["parent"]).rename(columns={"mu_mumu": "mu_parent", "chi2_mumu": "chi2_parent"})
    cand_df = pd.read_csv(paths["candidate"]).rename(columns={"mu_mumu": "mu_cf", "chi2_mumu": "chi2_cf"})
    merged = (
        full_df[["D", "eta", "mu_full", "chi2_full"]]
        .merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
        .merge(parent_df[["D", "eta", "mu_parent", "chi2_parent"]], on=["D", "eta"], how="inner")
        .merge(cand_df[["D", "eta", "mu_cf", "chi2_cf"]], on=["D", "eta"], how="inner")
        .sort_values(["D", "eta"])
        .reset_index(drop=True)
    )
    merged["case"] = case_name
    merged["abs_delta_base"] = (merged["mu_base"] - merged["mu_full"]).abs()
    merged["abs_delta_parent"] = (merged["mu_parent"] - merged["mu_full"]).abs()
    merged["abs_delta_cf"] = (merged["mu_cf"] - merged["mu_full"]).abs()
    merged["abs_improvement_vs_parent"] = merged["abs_delta_parent"] - merged["abs_delta_cf"]
    merged["acceptance_flip_vs_base"] = ((merged["chi2_base"] <= 4.0) ^ (merged["chi2_cf"] <= 4.0)).astype(float)
    return merged


def _monitor_mask(df: pd.DataFrame, subset: str, drift_d: float, neighbor_ds: list[float]) -> pd.Series:
    dvals = df["D"].to_numpy(dtype=float)
    reviewer_mask = np.zeros(len(df), dtype=bool)
    for target in D60_TARGETS:
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


def _summarize_subset(case_name: str, subset: str, df: pd.DataFrame, drift_d: float) -> dict[str, float | str | int]:
    worst_idx = df["abs_improvement_vs_parent"].idxmin()
    best_idx = df["abs_improvement_vs_parent"].idxmax()
    worst = df.loc[worst_idx]
    best = df.loc[best_idx]
    return {
        "case": case_name,
        "subset": subset,
        "count_points": int(len(df)),
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


def _slice_rows(case_name: str, df: pd.DataFrame, drift_d: float) -> pd.DataFrame:
    rows = []
    for d_val, grp in df.groupby("D", sort=True):
        d_float = float(d_val)
        rows.append({
            "case": case_name,
            "D": d_float,
            "is_reviewer_target": float(any(np.isclose(d_float, t, atol=1.0e-9) for t in D60_TARGETS)),
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
    active_cases: dict[str, dict[str, int]] = {}
    skipped_cases: dict[str, str] = {}
    for case_name, cfg in CASE_CONFIG.items():
        d_num = int(cfg["d_num"])
        eta_num = int(cfg["eta_num"])
        if not _full_direct_supported(d_num):
            skipped_cases[case_name] = (
                f"missing localized_direct_profiles_chi_Dgrid{d_num}.csv or "
                f"localized_direct_profiles_superrad_Dgrid{d_num}.csv"
            )
            continue
        tag_suffix = f"D{d_num}E{eta_num}_{TAG_SUFFIX}"
        active_cases[case_name] = {"d_num": d_num, "eta_num": eta_num}
        path_table[case_name] = {
            "full": _run_map("full_direct", f"chain_mode_full_direct_{tag_suffix}", d_num, eta_num),
            "base": _run_map(BASE_CHAIN_MODE, f"chain_mode_{BASE_CHAIN_MODE}_{tag_suffix}", d_num, eta_num),
            "parent": _run_map(PARENT_CHAIN_MODE, f"chain_mode_{PARENT_CHAIN_MODE}_{tag_suffix}", d_num, eta_num),
            "candidate": _run_map(CANDIDATE_CHAIN_MODE, f"chain_mode_{CANDIDATE_CHAIN_MODE}_{tag_suffix}", d_num, eta_num),
        }

    if not active_cases:
        raise RuntimeError("No monitoring cases are supported by the available full_direct localized profiles.")

    summary_rows = []
    slice_frames = []
    eta_frames = []
    detail_frames = []
    for case_name in active_cases:
        detail = _load_case(case_name, path_table[case_name])
        unique_ds = np.sort(detail["D"].unique().astype(float))
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
            sub = detail.loc[_monitor_mask(detail, subset, drift_d, neighbor_ds)].copy()
            summary_rows.append(_summarize_subset(case_name, subset, sub, drift_d))
        slice_frames.append(_slice_rows(case_name, detail, drift_d))
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
        "drift_target_D": DRIFT_TARGET_D,
        "d60_targets": D60_TARGETS,
        "active_cases": active_cases,
        "skipped_cases": skipped_cases,
        "paths": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in path_table.items()},
        "notes": "Production-style D60 monitoring focused on the residual d8comp non-reviewer micro-drift band near D≈7.254.",
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
