#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pslt_lib import PSLTKinetics
from scan_hll_signal_strengths import make_baseline_kinetics
from scan_prolate_hybrid_chain_propagation_audit import (
    CHAIN_MODE,
    D_REF,
    GN_PATH,
    GN_SPEC_PATH,
    N_MAX,
    OBSERVABLE_MODE,
    T_COH,
    _anchor_to_reference,
    _build_hybrid_profiles,
    _build_kinetics,
)
from scan_prolate_hybrid_reviewer_counterfactual_audit import REGIONS, _subset_rows

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
STEM = "prolate_parabolic_cloud_extraction_audit"

GRID_PROFILES = {
    "shortlist": {
        "peak": (0.5, 0.75, 1.0),
        "center": (5.9, 6.1),
        "width": (0.15, 0.25),
        "s_min": (0.55, 0.70),
        "power": (2.0,),
    },
    "full": {
        "peak": (0.25, 0.5, 0.75, 1.0),
        "center": (5.9, 6.1),
        "width": (0.15, 0.25),
        "s_min": (0.55, 0.70),
        "power": (1.0, 2.0, 3.0),
    },
}
CLOUD_SIDE = 9


def _gate(d_vals: np.ndarray, peak: float, center: float, width: float) -> np.ndarray:
    x = (np.asarray(d_vals, dtype=float) - float(center)) / max(float(width), 1e-9)
    return float(peak) / (1.0 + np.exp(-x))


def _cloud_average_profile(
    d_vals: np.ndarray,
    base_vals: np.ndarray,
    peak: float,
    center: float,
    width: float,
    s_min: float,
    power: float,
) -> np.ndarray:
    d_vals = np.asarray(d_vals, dtype=float)
    base_vals = np.asarray(base_vals, dtype=float)
    q = np.linspace(0.0, 1.0, CLOUD_SIDE)
    sigma = float(s_min) + (1.0 - float(s_min)) * (q**float(power))
    tau = float(s_min) + (1.0 - float(s_min)) * (q**float(power))
    scale = np.sqrt(np.outer(sigma, tau)).reshape(-1)
    weights = np.ones_like(scale)

    d_min = float(np.min(d_vals))
    d_max = float(np.max(d_vals))
    cloud = np.empty_like(base_vals, dtype=float)
    for i, D in enumerate(d_vals):
        local_d = np.clip(float(D) * scale, d_min, d_max)
        local_vals = np.interp(local_d, d_vals, base_vals)
        cloud[i] = float(np.average(local_vals, weights=weights))

    alpha = _gate(d_vals, peak=float(peak), center=float(center), width=float(width))
    mixed = (1.0 - alpha) * base_vals + alpha * cloud
    return _anchor_to_reference(d_vals, mixed, base_vals, ref_d=D_REF)


def _build_cloud_kinetics(runtime: PSLTKinetics, hybrid_detail: pd.DataFrame, tmpdir: Path, combo: dict[str, float]) -> tuple[PSLTKinetics, Path, Path]:
    d_vals = hybrid_detail["D"].to_numpy(dtype=float)
    chi_cloud = _cloud_average_profile(
        d_vals,
        hybrid_detail["chi_LR_hybrid"].to_numpy(dtype=float),
        combo["peak"],
        combo["center_D"],
        combo["width_D"],
        combo["s_min"],
        combo["power"],
    )
    g2_cloud = _cloud_average_profile(
        d_vals,
        hybrid_detail["g2_raw_hybrid"].to_numpy(dtype=float),
        combo["peak"],
        combo["center_D"],
        combo["width_D"],
        combo["s_min"],
        combo["power"],
    )
    g3_cloud = _cloud_average_profile(
        d_vals,
        hybrid_detail["g3_raw_hybrid"].to_numpy(dtype=float),
        combo["peak"],
        combo["center_D"],
        combo["width_D"],
        combo["s_min"],
        combo["power"],
    )

    gn_base = pd.read_csv(GN_PATH)
    if "level" in gn_base.columns:
        gn_base = gn_base.loc[gn_base["level"].astype(str).str.lower() == "fine"].copy()
    gn_base = gn_base.merge(
        pd.DataFrame({"D": d_vals, "g2_raw_cloud": g2_cloud, "g3_raw_cloud": g3_cloud}),
        on="D",
        how="inner",
    )
    gn_base["g2_raw"] = gn_base["g2_raw_cloud"]
    gn_base["g3_raw"] = gn_base["g3_raw_cloud"]
    gn_base["g1_hat"] = gn_base["g1_raw"] / np.maximum(gn_base["g3_raw"], 1e-30)
    gn_base["g2_hat"] = gn_base["g2_raw"] / np.maximum(gn_base["g3_raw"], 1e-30)
    gn_base["g3_hat"] = 1.0
    gn_base = gn_base.drop(columns=["g2_raw_cloud", "g3_raw_cloud"])

    scale_df = hybrid_detail[["D", "g2_raw", "g3_raw"]].copy()
    scale_df["g2_raw_cloud"] = g2_cloud
    scale_df["g3_raw_cloud"] = g3_cloud
    scale_df["scale2"] = scale_df["g2_raw_cloud"] / np.maximum(scale_df["g2_raw"], 1e-30)
    scale_df["scale3"] = scale_df["g3_raw_cloud"] / np.maximum(scale_df["g3_raw"], 1e-30)

    spec_base = pd.read_csv(GN_SPEC_PATH)
    if "level" in spec_base.columns:
        spec_base = spec_base.loc[spec_base["level"].astype(str).str.lower() == "fine"].copy()
    spec_base = spec_base.merge(scale_df[["D", "scale2", "scale3", "g3_raw_cloud"]], on="D", how="inner")

    mode_n = spec_base["mode_n"].astype(int).to_numpy()
    g_raw = spec_base["g_raw_n"].to_numpy(dtype=float)
    scale2 = spec_base["scale2"].to_numpy(dtype=float)
    scale3 = spec_base["scale3"].to_numpy(dtype=float)
    g3_cloud_ref = spec_base["g3_raw_cloud"].to_numpy(dtype=float)

    g_new = g_raw.copy()
    g_new[mode_n == 2] = g_raw[mode_n == 2] * scale2[mode_n == 2]
    g_new[mode_n >= 3] = g_raw[mode_n >= 3] * scale3[mode_n >= 3]
    spec_base["g_raw_n"] = np.maximum(g_new, 1e-30)
    spec_base["g_hat_to_g3"] = spec_base["g_raw_n"] / np.maximum(g3_cloud_ref, 1e-30)
    spec_base = spec_base.drop(columns=["scale2", "scale3", "g3_raw_cloud"])

    chi_df = pd.DataFrame({"D": d_vals, "chi_LR": chi_cloud})
    chi_path = tmpdir / f"cloud_chi_{combo['combo_id']:03d}.csv"
    gn_path = tmpdir / f"cloud_gn_{combo['combo_id']:03d}.csv"
    spec_path = tmpdir / f"cloud_spec_{combo['combo_id']:03d}.csv"
    chi_df.to_csv(chi_path, index=False)
    gn_base.to_csv(gn_path, index=False)
    spec_base.to_csv(spec_path, index=False)

    cloud_params = replace(
        runtime.params,
        chi_mode="localized_grid",
        chi_lr_D=tuple(float(x) for x in d_vals),
        chi_lr_vals=tuple(float(x) for x in chi_cloud),
        g_mode="fp_2d_full",
        g_fp_2d_csv=str(gn_path),
        g_fp_2d_spectrum_csv=str(spec_path),
    )
    return PSLTKinetics(params=cloud_params), chi_path, gn_path


def _region_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for region, cfg in REGIONS.items():
        full_df = _subset_rows(cfg["full_map"], cfg["D_targets"]).rename(columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"})
        base_df = _subset_rows(cfg["base_map"], cfg["D_targets"]).rename(columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"})
        hybrid_df = full_df.merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
        tables[region] = hybrid_df
    return tables


def _evaluate_combo(combo: dict[str, float], runtime: PSLTKinetics, hybrid_detail: pd.DataFrame, hybrid_kin: PSLTKinetics, region_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    slice_rows = []
    with tempfile.TemporaryDirectory(dir=OUTDIR) as tmp:
        cloud_kin, _, _ = _build_cloud_kinetics(runtime, hybrid_detail, Path(tmp), combo)
        for region, cfg in REGIONS.items():
            merged = region_tables[region]
            detail_rows = []
            for row in merged.itertuples(index=False):
                mu_hybrid = float(
                    hybrid_kin.hll_mu_pred(
                        2,
                        D=float(row.D),
                        eta=float(row.eta),
                        t_coh=T_COH,
                        ref_D=float(cfg["ref_D"]),
                        ref_eta=float(cfg["ref_eta"]),
                        observable_mode=OBSERVABLE_MODE,
                        N_max=N_MAX,
                    )
                )
                mu_cloud = float(
                    cloud_kin.hll_mu_pred(
                        2,
                        D=float(row.D),
                        eta=float(row.eta),
                        t_coh=T_COH,
                        ref_D=float(cfg["ref_D"]),
                        ref_eta=float(cfg["ref_eta"]),
                        observable_mode=OBSERVABLE_MODE,
                        N_max=N_MAX,
                    )
                )
                delta_base = float(row.mu_base - row.mu_full)
                delta_hybrid = float(mu_hybrid - row.mu_full)
                delta_cloud = float(mu_cloud - row.mu_full)
                detail_rows.append(
                    {
                        **combo,
                        "region": region,
                        "D": float(row.D),
                        "eta": float(row.eta),
                        "mu_full": float(row.mu_full),
                        "mu_base": float(row.mu_base),
                        "mu_hybrid": mu_hybrid,
                        "mu_cloud": mu_cloud,
                        "abs_delta_base": abs(delta_base),
                        "abs_delta_hybrid": abs(delta_hybrid),
                        "abs_delta_cloud": abs(delta_cloud),
                        "cloud_improvement_vs_base": abs(delta_base) - abs(delta_cloud),
                        "cloud_improvement_vs_hybrid": abs(delta_hybrid) - abs(delta_cloud),
                        "cloud_shift_vs_base": mu_cloud - row.mu_base,
                        "cloud_shift_vs_hybrid": mu_cloud - mu_hybrid,
                        "cloud_acceptance_flip_vs_base": float(((row.chi2_base <= 4.0) ^ ((((mu_cloud - 1.4) / 0.4) ** 2) <= 4.0))),
                    }
                )
            region_detail = pd.DataFrame(detail_rows)
            rows.append(
                {
                    **combo,
                    "region": region,
                    "p95_abs_delta_base": float(np.percentile(region_detail["abs_delta_base"], 95.0)),
                    "p95_abs_delta_hybrid": float(np.percentile(region_detail["abs_delta_hybrid"], 95.0)),
                    "p95_abs_delta_cloud": float(np.percentile(region_detail["abs_delta_cloud"], 95.0)),
                    "max_abs_delta_base": float(np.max(region_detail["abs_delta_base"])),
                    "max_abs_delta_hybrid": float(np.max(region_detail["abs_delta_hybrid"])),
                    "max_abs_delta_cloud": float(np.max(region_detail["abs_delta_cloud"])),
                    "mean_cloud_improvement_vs_base": float(np.mean(region_detail["cloud_improvement_vs_base"])),
                    "mean_cloud_improvement_vs_hybrid": float(np.mean(region_detail["cloud_improvement_vs_hybrid"])),
                    "improved_fraction_vs_base": float(np.mean(region_detail["cloud_improvement_vs_base"] > 0.0)),
                    "improved_fraction_vs_hybrid": float(np.mean(region_detail["cloud_improvement_vs_hybrid"] > 0.0)),
                    "p95_abs_cloud_shift_vs_base": float(np.percentile(np.abs(region_detail["cloud_shift_vs_base"]), 95.0)),
                    "acceptance_flip_fraction_vs_base": float(np.mean(region_detail["cloud_acceptance_flip_vs_base"])),
                }
            )
            for d_val, grp in region_detail.groupby("D", sort=True):
                slice_rows.append(
                    {
                        **combo,
                        "region": region,
                        "D": float(d_val),
                        "slice_p95_abs_delta_base": float(np.percentile(grp["abs_delta_base"], 95.0)),
                        "slice_p95_abs_delta_hybrid": float(np.percentile(grp["abs_delta_hybrid"], 95.0)),
                        "slice_p95_abs_delta_cloud": float(np.percentile(grp["abs_delta_cloud"], 95.0)),
                        "slice_mean_cloud_improvement_vs_base": float(np.mean(grp["cloud_improvement_vs_base"])),
                        "slice_mean_cloud_improvement_vs_hybrid": float(np.mean(grp["cloud_improvement_vs_hybrid"])),
                    }
                )
        return pd.DataFrame(rows), pd.DataFrame(slice_rows)


def _wide_summary(region_summary: pd.DataFrame) -> pd.DataFrame:
    d60 = region_summary.loc[region_summary["region"] == "D60_hotspot"].copy().set_index("combo_id")
    d21 = region_summary.loc[region_summary["region"] == "D21_guard"].copy().set_index("combo_id")
    rows = []
    for combo_id in d60.index:
        r60 = d60.loc[combo_id]
        r21 = d21.loc[combo_id]
        rows.append(
            {
                "combo_id": int(combo_id),
                "peak": float(r60["peak"]),
                "center_D": float(r60["center_D"]),
                "width_D": float(r60["width_D"]),
                "s_min": float(r60["s_min"]),
                "power": float(r60["power"]),
                "D60_p95_base": float(r60["p95_abs_delta_base"]),
                "D60_p95_hybrid": float(r60["p95_abs_delta_hybrid"]),
                "D60_p95_cloud": float(r60["p95_abs_delta_cloud"]),
                "D21_p95_base": float(r21["p95_abs_delta_base"]),
                "D21_p95_hybrid": float(r21["p95_abs_delta_hybrid"]),
                "D21_p95_cloud": float(r21["p95_abs_delta_cloud"]),
                "D60_mean_improvement_vs_base": float(r60["mean_cloud_improvement_vs_base"]),
                "D60_mean_improvement_vs_hybrid": float(r60["mean_cloud_improvement_vs_hybrid"]),
                "D21_mean_improvement_vs_base": float(r21["mean_cloud_improvement_vs_base"]),
                "D21_mean_improvement_vs_hybrid": float(r21["mean_cloud_improvement_vs_hybrid"]),
                "D60_improved_fraction_vs_base": float(r60["improved_fraction_vs_base"]),
                "D21_improved_fraction_vs_base": float(r21["improved_fraction_vs_base"]),
                "D21_acceptance_flip_fraction": float(r21["acceptance_flip_fraction_vs_base"]),
                "joint_objective": float(r60["p95_abs_delta_cloud"] + 8.0 * max(float(r21["p95_abs_delta_cloud"]) - float(r21["p95_abs_delta_base"]), 0.0)),
                "net_mean_gain_vs_base": float(r60["mean_cloud_improvement_vs_base"] + r21["mean_cloud_improvement_vs_base"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["joint_objective", "D60_p95_cloud", "D21_p95_cloud"]).reset_index(drop=True)


def _plot(summary_wide: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
    sc = axes[0].scatter(
        summary_wide["D60_p95_cloud"],
        summary_wide["D21_p95_cloud"],
        c=summary_wide["peak"],
        cmap="viridis",
        s=50,
        edgecolor="k",
        linewidth=0.3,
    )
    axes[0].set_title("cloud p95: D60 vs D21")
    axes[0].set_xlabel("D60 hotspot p95 |Δμ|")
    axes[0].set_ylabel("D21 guard p95 |Δμ|")
    axes[1].scatter(summary_wide["D60_mean_improvement_vs_base"], summary_wide["D21_mean_improvement_vs_base"], s=46, color="#e15759")
    axes[1].set_title("Mean improvement vs base")
    axes[1].set_xlabel("D60 mean improvement")
    axes[1].set_ylabel("D21 mean improvement")
    axes[2].plot(summary_wide.index, summary_wide["joint_objective"], marker="o", lw=1.5)
    axes[2].set_title("Ranked joint objective")
    axes[2].set_xlabel("sorted combo rank")
    axes[2].set_ylabel("objective")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=axes[0], shrink=0.85, label="peak")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(GRID_PROFILES.keys()), default="shortlist")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    hybrid_detail, chi_path, gn_path, spec_path = _build_hybrid_profiles()
    d_grid = hybrid_detail["D"].to_numpy(dtype=float)
    base_kinetics = _build_kinetics(chi_path, gn_path, spec_path, hybrid_detail)
    runtime = base_kinetics["runtime_mainline"]
    hybrid_kin = base_kinetics["hybrid_profile"]
    region_tables = _region_tables()

    grid = GRID_PROFILES[args.profile]
    total = (
        len(grid["peak"])
        * len(grid["center"])
        * len(grid["width"])
        * len(grid["s_min"])
        * len(grid["power"])
    )

    all_region_rows = []
    all_slice_rows = []
    combos = []
    combo_id = 0
    for peak in grid["peak"]:
        for center in grid["center"]:
            for width in grid["width"]:
                for s_min in grid["s_min"]:
                    for power in grid["power"]:
                        combo_id += 1
                        combo = {
                            "combo_id": combo_id,
                            "peak": float(peak),
                            "center_D": float(center),
                            "width_D": float(width),
                            "s_min": float(s_min),
                            "power": float(power),
                        }
                        combos.append(combo)
                        region_summary, slice_summary = _evaluate_combo(combo, runtime, hybrid_detail, hybrid_kin, region_tables)
                        all_region_rows.append(region_summary)
                        all_slice_rows.append(slice_summary)
                        print(
                            f"[combo {combo_id}/{total}] peak={peak} center={center} width={width} "
                            f"s_min={s_min} power={power}",
                            flush=True,
                        )

    region_summary = pd.concat(all_region_rows, ignore_index=True)
    slices = pd.concat(all_slice_rows, ignore_index=True)
    summary_wide = _wide_summary(region_summary)

    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    detail_path = OUTDIR / f"{STEM}_detail.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    summary_wide.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    region_summary.to_csv(detail_path, index=False)
    _plot(summary_wide, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "combos": combos,
                "grid_profile": args.profile,
                "cloud_side": CLOUD_SIDE,
                "notes": (
                    "High-D parabolic cloud audit: keeps Gamma_ref on uniform/runtime baseline and only applies a gated cloud-average "
                    "to the calibrated prolate chi_LR / g2_raw / g3_raw extraction family. Goal is to reduce D60 hotspot residuals "
                    "without reopening D21 guard."
                ),
            },
            indent=2,
        )
    )

    for path in [summary_path, slices_path, detail_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(summary_wide.head(12).to_string(index=False))
    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")
    print(f"Wrote detail:  {detail_path}")


if __name__ == "__main__":
    main()
