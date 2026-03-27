#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_prolate_hybrid_chain_propagation_audit import (
    N_MAX,
    OBSERVABLE_MODE,
    T_COH,
    _build_hybrid_profiles,
    _build_kinetics,
)


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
STEM = "prolate_hybrid_reviewer_counterfactual_audit"
MAP_DIR = ROOT / "output" / "hll_signal_strength"

REGIONS = {
    "D60_hotspot": {
        "full_map": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D60E21_refresh_20260324.csv",
        "base_map": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D60E21_refresh_20260324.csv",
        "ref_D": 9.966101694915254,
        "ref_eta": 1.0,
        "D_targets": [
            4.813559322033898,
            5.627118644067797,
            6.169492525423729,
            6.4406779661016955,
            6.711864406779661,
        ],
    },
    "D21_guard": {
        "full_map": MAP_DIR / "hll_signal_strength_map_chain_mode_full_direct_D21E41_refresh_20260324.csv",
        "base_map": MAP_DIR / "hll_signal_strength_map_chain_mode_cell_direct_runtime_release_fullwidthrefamp_pointamp2_widthboost_D21E41.csv",
        "ref_D": 9.6,
        "ref_eta": 1.0,
        "D_targets": [4.0, 6.4, 8.0],
    },
}


def _subset_rows(path: Path, targets: list[float]) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = np.zeros(len(df), dtype=bool)
    dvals = df["D"].to_numpy(dtype=float)
    for target in targets:
        mask |= np.isclose(dvals, float(target), atol=1e-9)
    return df.loc[mask].copy().sort_values(["D", "eta"]).reset_index(drop=True)


def _detail_rows(hybrid_kin) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for region, cfg in REGIONS.items():
        full_df = _subset_rows(cfg["full_map"], cfg["D_targets"]).rename(
            columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"}
        )
        base_df = _subset_rows(cfg["base_map"], cfg["D_targets"]).rename(
            columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"}
        )
        merged = full_df.merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")

        for row in merged.itertuples(index=False):
            mu_cf = float(
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
            chi2_cf = float(((mu_cf - 1.4) / 0.4) ** 2)
            delta_base = float(row.mu_base - row.mu_full)
            delta_cf = float(mu_cf - row.mu_full)
            rows.append(
                {
                    "region": region,
                    "D": float(row.D),
                    "eta": float(row.eta),
                    "mu_full": float(row.mu_full),
                    "mu_base": float(row.mu_base),
                    "mu_cf": mu_cf,
                    "chi2_full": float(row.chi2_full),
                    "chi2_base": float(row.chi2_base),
                    "chi2_cf": chi2_cf,
                    "delta_base": delta_base,
                    "delta_cf": delta_cf,
                    "abs_delta_base": float(abs(delta_base)),
                    "abs_delta_cf": float(abs(delta_cf)),
                    "abs_improvement": float(abs(delta_base) - abs(delta_cf)),
                    "cf_shift_vs_base": float(mu_cf - row.mu_base),
                    "improved": float(abs(delta_cf) < abs(delta_base)),
                    "worsened": float(abs(delta_cf) > abs(delta_base)),
                    "obs_acceptance_flip_vs_base": float((row.chi2_base <= 4.0) ^ (chi2_cf <= 4.0)),
                }
            )
    return pd.DataFrame(rows).sort_values(["region", "D", "eta"]).reset_index(drop=True)


def _summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for region, grp in detail.groupby("region", sort=False):
        rows.append(
            {
                "region": region,
                "count_points": int(len(grp)),
                "p95_abs_delta_base": float(np.percentile(grp["abs_delta_base"], 95.0)),
                "p95_abs_delta_cf": float(np.percentile(grp["abs_delta_cf"], 95.0)),
                "max_abs_delta_base": float(np.max(grp["abs_delta_base"])),
                "max_abs_delta_cf": float(np.max(grp["abs_delta_cf"])),
                "mean_abs_improvement": float(np.mean(grp["abs_improvement"])),
                "median_abs_improvement": float(np.median(grp["abs_improvement"])),
                "improved_fraction": float(np.mean(grp["improved"])),
                "worsened_fraction": float(np.mean(grp["worsened"])),
                "mean_cf_shift_vs_base": float(np.mean(grp["cf_shift_vs_base"])),
                "p95_abs_cf_shift_vs_base": float(np.percentile(np.abs(grp["cf_shift_vs_base"]), 95.0)),
                "obs_acceptance_flip_fraction": float(np.mean(grp["obs_acceptance_flip_vs_base"])),
            }
        )
    return pd.DataFrame(rows)


def _slices(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (region, d_val), grp in detail.groupby(["region", "D"], sort=False):
        rows.append(
            {
                "region": region,
                "D": float(d_val),
                "slice_p95_abs_delta_base": float(np.percentile(grp["abs_delta_base"], 95.0)),
                "slice_p95_abs_delta_cf": float(np.percentile(grp["abs_delta_cf"], 95.0)),
                "slice_max_abs_delta_base": float(np.max(grp["abs_delta_base"])),
                "slice_max_abs_delta_cf": float(np.max(grp["abs_delta_cf"])),
                "slice_mean_abs_improvement": float(np.mean(grp["abs_improvement"])),
                "slice_improved_fraction": float(np.mean(grp["improved"])),
                "slice_worsened_fraction": float(np.mean(grp["worsened"])),
                "slice_p95_abs_cf_shift_vs_base": float(np.percentile(np.abs(grp["cf_shift_vs_base"]), 95.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["region", "D"]).reset_index(drop=True)


def _plot(slices: pd.DataFrame, out_png: Path) -> None:
    colors = {"D60_hotspot": "#e15759", "D21_guard": "#4e79a7"}
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), sharex=False)
    for region, grp in slices.groupby("region", sort=False):
        color = colors[region]
        axes[0].plot(grp["D"], grp["slice_p95_abs_delta_base"], marker="o", ls="--", color=color, alpha=0.65, label=f"{region} base")
        axes[0].plot(grp["D"], grp["slice_p95_abs_delta_cf"], marker="o", color=color, label=f"{region} cf")
        axes[1].plot(grp["D"], grp["slice_mean_abs_improvement"], marker="o", color=color, label=region)
        axes[2].plot(grp["D"], grp["slice_improved_fraction"], marker="o", color=color, label=region)
    axes[0].set_title("p95 |Δμ| vs full_direct")
    axes[1].set_title("Mean |Δ| improvement")
    axes[2].set_title("Improved fraction")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("D")
    axes[0].set_ylabel("value")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    hybrid_detail, chi_path, gn_path, spec_path = _build_hybrid_profiles()
    kinetics = _build_kinetics(chi_path, gn_path, spec_path, hybrid_detail)
    hybrid_kin = kinetics["hybrid_profile"]

    detail = _detail_rows(hybrid_kin)
    summary = _summary(detail)
    slices = _slices(detail)

    detail_path = OUTDIR / f"{STEM}_detail.csv"
    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(slices, png_path)
    meta_path.write_text(
        json.dumps(
            {
                "observable_mode": OBSERVABLE_MODE,
                "regions": {
                    name: {
                        "full_map": str(cfg["full_map"]),
                        "base_map": str(cfg["base_map"]),
                        "ref_D": float(cfg["ref_D"]),
                        "ref_eta": float(cfg["ref_eta"]),
                        "D_targets": [float(x) for x in cfg["D_targets"]],
                    }
                    for name, cfg in REGIONS.items()
                },
                "hybrid_source_profiles": {
                    "chi_profile": str(chi_path),
                    "gn_profile": str(gn_path),
                    "gn_spectrum": str(spec_path),
                },
                "notes": (
                    "Counterfactual reviewer audit keeps the current mainline map unchanged and only evaluates how the "
                    "hybrid prolate chi/g observable effect would shift mu_mumu on refreshed D60 hotspots and D21 guards."
                ),
            },
            indent=2,
        )
    )

    for path in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(summary.to_string(index=False))
    print(f"\nWrote detail:  {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")


if __name__ == "__main__":
    main()
