#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_prolate_hybrid_reviewer_counterfactual_audit import REGIONS, _subset_rows


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"
STEM = "gamma_ref_highd_counterfactual_audit"

MODE_GRID = ("gamma_ref_enhance", "gamma_ref_suppress")
KAPPA_GRID = (0.05, 0.10, 0.15, 0.20)
CENTER_GRID = (6.10, 6.25, 6.40, 6.55)
WIDTH_GRID = (0.12, 0.20, 0.30)
POWER_GRID = (0.5,)


def _gate(D: float, center: float, width: float) -> float:
    x = (float(D) - float(center)) / max(float(width), 1e-9)
    return float(1.0 / (1.0 + np.exp(-x)))


def _counterfactual_mu(mu_base: float, D: float, mode: str, kappa: float, center_D: float, width_D: float, power: float) -> float:
    gate = _gate(D, center=float(center_D), width=float(width_D))
    scale = float(max(float(D), 0.0) ** float(power))
    exponent = float(kappa) * gate * scale
    if mode == "gamma_ref_enhance":
        # Larger Gamma_ref should damp chi/partial-like observables.
        factor = float(np.exp(-exponent))
    elif mode == "gamma_ref_suppress":
        # Smaller Gamma_ref moves in the opposite direction and tests the sign
        # ambiguity in the informal derivation.
        factor = float(np.exp(+exponent))
    else:
        raise ValueError(f"unknown mode: {mode}")
    return float(mu_base * factor)


def _region_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for region, cfg in REGIONS.items():
        full_df = _subset_rows(cfg["full_map"], cfg["D_targets"]).rename(
            columns={"mu_mumu": "mu_full", "chi2_mumu": "chi2_full"}
        )
        base_df = _subset_rows(cfg["base_map"], cfg["D_targets"]).rename(
            columns={"mu_mumu": "mu_base", "chi2_mumu": "chi2_base"}
        )
        tables[region] = full_df.merge(base_df[["D", "eta", "mu_base", "chi2_base"]], on=["D", "eta"], how="inner")
    return tables


def _evaluate_combo(combo: dict[str, float | str], region_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_rows = []
    slice_rows = []
    for region, merged in region_tables.items():
        detail = []
        for row in merged.itertuples(index=False):
            mu_cf = _counterfactual_mu(
                mu_base=float(row.mu_base),
                D=float(row.D),
                mode=str(combo["mode"]),
                kappa=float(combo["kappa"]),
                center_D=float(combo["center_D"]),
                width_D=float(combo["width_D"]),
                power=float(combo["power"]),
            )
            delta_base = float(row.mu_base - row.mu_full)
            delta_cf = float(mu_cf - row.mu_full)
            chi2_cf = float(((mu_cf - 1.4) / 0.4) ** 2)
            detail.append(
                {
                    **combo,
                    "region": region,
                    "D": float(row.D),
                    "eta": float(row.eta),
                    "mu_full": float(row.mu_full),
                    "mu_base": float(row.mu_base),
                    "mu_cf": mu_cf,
                    "chi2_base": float(row.chi2_base),
                    "chi2_cf": chi2_cf,
                    "abs_delta_base": abs(delta_base),
                    "abs_delta_cf": abs(delta_cf),
                    "abs_improvement": abs(delta_base) - abs(delta_cf),
                    "cf_shift_vs_base": float(mu_cf - row.mu_base),
                    "improved": float(abs(delta_cf) < abs(delta_base)),
                    "worsened": float(abs(delta_cf) > abs(delta_base)),
                    "acceptance_flip_vs_base": float(((row.chi2_base <= 4.0) ^ (chi2_cf <= 4.0))),
                    "gate": _gate(float(row.D), float(combo["center_D"]), float(combo["width_D"])),
                }
            )
        region_detail = pd.DataFrame(detail)
        region_rows.append(
            {
                **combo,
                "region": region,
                "p95_abs_delta_base": float(np.percentile(region_detail["abs_delta_base"], 95.0)),
                "p95_abs_delta_cf": float(np.percentile(region_detail["abs_delta_cf"], 95.0)),
                "max_abs_delta_base": float(np.max(region_detail["abs_delta_base"])),
                "max_abs_delta_cf": float(np.max(region_detail["abs_delta_cf"])),
                "mean_abs_improvement": float(np.mean(region_detail["abs_improvement"])),
                "improved_fraction": float(np.mean(region_detail["improved"])),
                "worsened_fraction": float(np.mean(region_detail["worsened"])),
                "p95_abs_cf_shift_vs_base": float(np.percentile(np.abs(region_detail["cf_shift_vs_base"]), 95.0)),
                "acceptance_flip_fraction": float(np.mean(region_detail["acceptance_flip_vs_base"])),
            }
        )
        for d_val, grp in region_detail.groupby("D", sort=True):
            slice_rows.append(
                {
                    **combo,
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
    return pd.DataFrame(region_rows), pd.DataFrame(slice_rows)


def _plot(slices: pd.DataFrame, out_png: Path) -> None:
    top = (
        slices.groupby(["mode", "kappa", "center_D", "width_D", "power", "region"], as_index=False)["slice_p95_abs_delta_cf"]
        .mean()
        .pivot_table(
            index=["mode", "kappa", "center_D", "width_D", "power"],
            columns="region",
            values="slice_p95_abs_delta_cf",
        )
        .reset_index()
    )
    if top.empty:
        return
    top["joint"] = np.maximum(top["D60_hotspot"], top["D21_guard"])
    best = top.sort_values(["joint", "D60_hotspot", "D21_guard"]).iloc[0]
    mask = (
        (slices["mode"] == best["mode"])
        & np.isclose(slices["kappa"], best["kappa"])
        & np.isclose(slices["center_D"], best["center_D"])
        & np.isclose(slices["width_D"], best["width_D"])
        & np.isclose(slices["power"], best["power"])
    )
    grp = slices.loc[mask].copy()

    colors = {"D60_hotspot": "#e15759", "D21_guard": "#4e79a7"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=False)
    for region, region_grp in grp.groupby("region", sort=False):
        color = colors[region]
        axes[0].plot(region_grp["D"], region_grp["slice_p95_abs_delta_base"], marker="o", ls="--", color=color, alpha=0.65, label=f"{region} base")
        axes[0].plot(region_grp["D"], region_grp["slice_p95_abs_delta_cf"], marker="o", color=color, label=f"{region} cf")
        axes[1].plot(region_grp["D"], region_grp["slice_mean_abs_improvement"], marker="o", color=color, label=region)
        axes[2].plot(region_grp["D"], region_grp["slice_improved_fraction"], marker="o", color=color, label=region)
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

    region_tables = _region_tables()
    summary_rows = []
    slice_rows = []
    detail_rows = []

    combo_id = 0
    total = len(MODE_GRID) * len(KAPPA_GRID) * len(CENTER_GRID) * len(WIDTH_GRID) * len(POWER_GRID)
    for mode in MODE_GRID:
        for kappa in KAPPA_GRID:
            for center_D in CENTER_GRID:
                for width_D in WIDTH_GRID:
                    for power in POWER_GRID:
                        combo_id += 1
                        print(f"[combo {combo_id}/{total}] mode={mode} kappa={kappa:.3f} center={center_D:.3f} width={width_D:.3f} power={power:.3f}")
                        combo = {
                            "combo_id": combo_id,
                            "mode": mode,
                            "kappa": float(kappa),
                            "center_D": float(center_D),
                            "width_D": float(width_D),
                            "power": float(power),
                        }
                        region_summary, region_slices = _evaluate_combo(combo, region_tables)
                        summary_rows.append(region_summary)
                        slice_rows.append(region_slices)
                        for region, merged in region_tables.items():
                            for row in merged.itertuples(index=False):
                                mu_cf = _counterfactual_mu(
                                    mu_base=float(row.mu_base),
                                    D=float(row.D),
                                    mode=mode,
                                    kappa=float(kappa),
                                    center_D=float(center_D),
                                    width_D=float(width_D),
                                    power=float(power),
                                )
                                detail_rows.append(
                                    {
                                        **combo,
                                        "region": region,
                                        "D": float(row.D),
                                        "eta": float(row.eta),
                                        "mu_full": float(row.mu_full),
                                        "mu_base": float(row.mu_base),
                                        "mu_cf": mu_cf,
                                        "abs_delta_base": abs(float(row.mu_base) - float(row.mu_full)),
                                        "abs_delta_cf": abs(float(mu_cf) - float(row.mu_full)),
                                        "abs_improvement": abs(float(row.mu_base) - float(row.mu_full))
                                        - abs(float(mu_cf) - float(row.mu_full)),
                                    }
                                )

    summary = pd.concat(summary_rows, ignore_index=True)
    slices = pd.concat(slice_rows, ignore_index=True)
    detail = pd.DataFrame(detail_rows)

    wide = (
        summary.pivot_table(
            index=["combo_id", "mode", "kappa", "center_D", "width_D", "power"],
            columns="region",
            values=["p95_abs_delta_base", "p95_abs_delta_cf", "mean_abs_improvement", "improved_fraction"],
        )
        .reset_index()
    )
    wide.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_") if isinstance(col, tuple) else str(col)
        for col in wide.columns
    ]
    wide["joint_objective"] = np.maximum(wide["p95_abs_delta_cf_D60_hotspot"], wide["p95_abs_delta_cf_D21_guard"])
    wide = wide.sort_values(
        ["joint_objective", "p95_abs_delta_cf_D60_hotspot", "p95_abs_delta_cf_D21_guard", "combo_id"]
    ).reset_index(drop=True)

    detail_path = OUTDIR / f"{STEM}_detail.csv"
    summary_path = OUTDIR / f"{STEM}_summary.csv"
    slices_path = OUTDIR / f"{STEM}_slices.csv"
    png_path = OUTDIR / f"{STEM}.png"
    meta_path = OUTDIR / f"{STEM}_run_meta.json"

    detail.to_csv(detail_path, index=False)
    wide.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    _plot(slices, png_path)
    meta_path.write_text(
        json.dumps(
            {
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
                "scan": {
                    "mode": list(MODE_GRID),
                    "kappa": list(KAPPA_GRID),
                    "center_D": list(CENTER_GRID),
                    "width_D": list(WIDTH_GRID),
                    "power": list(POWER_GRID),
                },
                "notes": (
                    "Reviewer-level counterfactual for a Gamma_ref-like high-D gate. "
                    "The current mainline does not route through open-system Gamma_ref, so this audit treats "
                    "Gamma_ref enhancement/suppression as a direct multiplicative denominator-like correction on mu."
                ),
            },
            indent=2,
        )
    )

    for path in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(wide.head(10).to_string(index=False))
    print(f"\nWrote detail:  {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")


if __name__ == "__main__":
    main()
