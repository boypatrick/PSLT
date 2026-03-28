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
STEM = "lorentz_helical_counterfactual_audit"

INPUT_DETAIL = OUTDIR / "uniform_lorentz_hybrid_reviewer_counterfactual_audit_detail.csv"
INPUT_SUMMARY = OUTDIR / "uniform_lorentz_hybrid_reviewer_counterfactual_audit_summary.csv"

BETA = 0.4
OMEGA_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)
M_GRID = (1.0, 2.0, 4.0, 8.0)
PHASE_GRID = (0.0, float(np.pi / 4.0), float(np.pi / 2.0), float(3.0 * np.pi / 4.0))
CENTER_GRID = (5.90, 6.10, 6.30)
WIDTH_GRID = (0.05, 0.10, 0.20)
ETA_COUPLE_GRID = (0.0, 0.5, 1.0)


def _gamma(beta: float) -> float:
    return float(1.0 / np.sqrt(max(1.0 - float(beta) ** 2, 1e-12)))


def _gate(D_eff: float, center_D: float, width_D: float) -> float:
    x = (float(D_eff) - float(center_D)) / max(float(width_D), 1e-9)
    return float(1.0 / (1.0 + np.exp(-x)))


def _helical_factor(D: float, eta: float, omega: float, m: float, phase: float, center_D: float, width_D: float, eta_couple: float) -> tuple[float, float, float]:
    D_eff = float(D) / _gamma(BETA)
    gate = _gate(D_eff, center_D=float(center_D), width_D=float(width_D))
    eta_phase = float(eta_couple) * np.pi * float(eta) / 4.0
    notch = 0.5 * (1.0 - np.cos(float(m) * (D_eff - float(center_D)) + float(phase) + eta_phase))
    factor = float(np.exp(-float(omega) * gate * notch))
    return D_eff, float(notch), factor


def _evaluate_combo(combo: dict[str, float], base_detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    slice_rows: list[dict[str, float | str]] = []

    for row in base_detail.itertuples(index=False):
        D_eff, notch, factor = _helical_factor(
            D=float(row.D),
            eta=float(row.eta),
            omega=float(combo["omega"]),
            m=float(combo["m"]),
            phase=float(combo["phase"]),
            center_D=float(combo["center_D"]),
            width_D=float(combo["width_D"]),
            eta_couple=float(combo["eta_couple"]),
        )
        mu_cf = float(row.mu_cf) * factor
        chi2_cf = float(((mu_cf - 1.4) / 0.4) ** 2)
        delta_mainline = float(row.mu_base - row.mu_full)
        delta_lorentz = float(row.mu_cf - row.mu_full)
        delta_cf = float(mu_cf - row.mu_full)
        detail_rows.append(
            {
                **combo,
                "region": str(row.region),
                "D": float(row.D),
                "eta": float(row.eta),
                "D_eff": D_eff,
                "gate": _gate(D_eff, combo["center_D"], combo["width_D"]),
                "helical_notch": notch,
                "helical_factor": factor,
                "mu_full": float(row.mu_full),
                "mu_mainline": float(row.mu_base),
                "mu_lorentz": float(row.mu_cf),
                "mu_cf": mu_cf,
                "chi2_mainline": float(row.chi2_base),
                "chi2_lorentz": float(row.chi2_cf),
                "chi2_cf": chi2_cf,
                "abs_delta_mainline": abs(delta_mainline),
                "abs_delta_lorentz": abs(delta_lorentz),
                "abs_delta_cf": abs(delta_cf),
                "abs_improvement_vs_mainline": abs(delta_mainline) - abs(delta_cf),
                "abs_improvement_vs_lorentz": abs(delta_lorentz) - abs(delta_cf),
                "cf_shift_vs_mainline": float(mu_cf - row.mu_base),
                "cf_shift_vs_lorentz": float(mu_cf - row.mu_cf),
                "improved_vs_mainline": float(abs(delta_cf) < abs(delta_mainline)),
                "improved_vs_lorentz": float(abs(delta_cf) < abs(delta_lorentz)),
                "worsened_vs_mainline": float(abs(delta_cf) > abs(delta_mainline)),
                "worsened_vs_lorentz": float(abs(delta_cf) > abs(delta_lorentz)),
            }
        )

    detail = pd.DataFrame(detail_rows).sort_values(["region", "D", "eta"]).reset_index(drop=True)

    for region, grp in detail.groupby("region", sort=False):
        summary_rows.append(
            {
                **combo,
                "region": region,
                "p95_abs_delta_mainline": float(np.percentile(grp["abs_delta_mainline"], 95.0)),
                "p95_abs_delta_lorentz": float(np.percentile(grp["abs_delta_lorentz"], 95.0)),
                "p95_abs_delta_cf": float(np.percentile(grp["abs_delta_cf"], 95.0)),
                "max_abs_delta_mainline": float(np.max(grp["abs_delta_mainline"])),
                "max_abs_delta_lorentz": float(np.max(grp["abs_delta_lorentz"])),
                "max_abs_delta_cf": float(np.max(grp["abs_delta_cf"])),
                "mean_abs_improvement_vs_mainline": float(np.mean(grp["abs_improvement_vs_mainline"])),
                "mean_abs_improvement_vs_lorentz": float(np.mean(grp["abs_improvement_vs_lorentz"])),
                "improved_fraction_vs_mainline": float(np.mean(grp["improved_vs_mainline"])),
                "improved_fraction_vs_lorentz": float(np.mean(grp["improved_vs_lorentz"])),
                "worsened_fraction_vs_mainline": float(np.mean(grp["worsened_vs_mainline"])),
                "worsened_fraction_vs_lorentz": float(np.mean(grp["worsened_vs_lorentz"])),
                "p95_abs_cf_shift_vs_mainline": float(np.percentile(np.abs(grp["cf_shift_vs_mainline"]), 95.0)),
                "p95_abs_cf_shift_vs_lorentz": float(np.percentile(np.abs(grp["cf_shift_vs_lorentz"]), 95.0)),
            }
        )
        for d_val, dgrp in grp.groupby("D", sort=True):
            slice_rows.append(
                {
                    **combo,
                    "region": region,
                    "D": float(d_val),
                    "slice_p95_abs_delta_mainline": float(np.percentile(dgrp["abs_delta_mainline"], 95.0)),
                    "slice_p95_abs_delta_lorentz": float(np.percentile(dgrp["abs_delta_lorentz"], 95.0)),
                    "slice_p95_abs_delta_cf": float(np.percentile(dgrp["abs_delta_cf"], 95.0)),
                    "slice_mean_abs_improvement_vs_mainline": float(np.mean(dgrp["abs_improvement_vs_mainline"])),
                    "slice_mean_abs_improvement_vs_lorentz": float(np.mean(dgrp["abs_improvement_vs_lorentz"])),
                    "slice_improved_fraction_vs_mainline": float(np.mean(dgrp["improved_vs_mainline"])),
                    "slice_improved_fraction_vs_lorentz": float(np.mean(dgrp["improved_vs_lorentz"])),
                    "slice_worsened_fraction_vs_mainline": float(np.mean(dgrp["worsened_vs_mainline"])),
                    "slice_worsened_fraction_vs_lorentz": float(np.mean(dgrp["worsened_vs_lorentz"])),
                }
            )

    return detail, pd.DataFrame(summary_rows), pd.DataFrame(slice_rows)


def _plot(slices: pd.DataFrame, out_png: Path) -> None:
    wide = (
        slices.pivot_table(
            index=["combo_id", "omega", "m", "phase", "center_D", "width_D", "eta_couple"],
            columns="region",
            values="slice_p95_abs_delta_cf",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide["joint"] = np.maximum(wide["D60_hotspot"], wide["D21_guard"])
    best = wide.sort_values(["joint", "D60_hotspot", "D21_guard"]).iloc[0]
    mask = (
        np.isclose(slices["combo_id"], best["combo_id"])
    )
    grp = slices.loc[mask].copy()
    colors = {"D60_hotspot": "#e15759", "D21_guard": "#4e79a7"}
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), sharex=False)
    for region, region_grp in grp.groupby("region", sort=False):
        color = colors[region]
        axes[0].plot(region_grp["D"], region_grp["slice_p95_abs_delta_mainline"], marker="o", ls="--", color=color, alpha=0.55, label=f"{region} mainline")
        axes[0].plot(region_grp["D"], region_grp["slice_p95_abs_delta_lorentz"], marker="s", ls=":", color=color, alpha=0.8, label=f"{region} lorentz")
        axes[0].plot(region_grp["D"], region_grp["slice_p95_abs_delta_cf"], marker="o", color=color, label=f"{region} helical")
        axes[1].plot(region_grp["D"], region_grp["slice_mean_abs_improvement_vs_mainline"], marker="o", color=color, label=f"{region} vs mainline")
        axes[2].plot(region_grp["D"], region_grp["slice_mean_abs_improvement_vs_lorentz"], marker="o", color=color, label=f"{region} vs lorentz")
    axes[0].set_title("p95 |Δμ| vs full_direct")
    axes[1].set_title("Mean |Δ| improvement vs mainline")
    axes[2].set_title("Mean |Δ| improvement vs Lorentz")
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

    base_detail = pd.read_csv(INPUT_DETAIL)
    detail_frames = []
    summary_frames = []
    slice_frames = []

    combos = [
        (omega, m, phase, center_D, width_D, eta_couple)
        for omega in OMEGA_GRID
        for m in M_GRID
        for phase in PHASE_GRID
        for center_D in CENTER_GRID
        for width_D in WIDTH_GRID
        for eta_couple in ETA_COUPLE_GRID
    ]
    for idx, (omega, m, phase, center_D, width_D, eta_couple) in enumerate(combos, start=1):
        print(
            f"[combo {idx}/{len(combos)}] omega={omega:.2f} m={m:.2f} phase={phase:.3f} center={center_D:.2f} width={width_D:.2f} eta_couple={eta_couple:.2f}",
            flush=True,
        )
        combo = {
            "combo_id": idx,
            "beta": float(BETA),
            "omega": float(omega),
            "m": float(m),
            "phase": float(phase),
            "center_D": float(center_D),
            "width_D": float(width_D),
            "eta_couple": float(eta_couple),
        }
        detail_df, summary_df, slices_df = _evaluate_combo(combo, base_detail)
        detail_frames.append(detail_df)
        summary_frames.append(summary_df)
        slice_frames.append(slices_df)

    detail = pd.concat(detail_frames, ignore_index=True)
    summary_long = pd.concat(summary_frames, ignore_index=True)
    slices = pd.concat(slice_frames, ignore_index=True)

    summary = (
        summary_long.pivot_table(
            index=["combo_id", "beta", "omega", "m", "phase", "center_D", "width_D", "eta_couple"],
            columns="region",
            values=[
                "p95_abs_delta_mainline",
                "p95_abs_delta_lorentz",
                "p95_abs_delta_cf",
                "mean_abs_improvement_vs_mainline",
                "mean_abs_improvement_vs_lorentz",
                "improved_fraction_vs_mainline",
                "improved_fraction_vs_lorentz",
            ],
        )
        .reset_index()
    )
    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary["joint_objective_vs_mainline"] = np.maximum(
        summary["p95_abs_delta_cf_D60_hotspot"],
        summary["p95_abs_delta_cf_D21_guard"],
    )
    summary["joint_objective_vs_lorentz"] = np.maximum(
        summary["p95_abs_delta_cf_D60_hotspot"],
        summary["p95_abs_delta_cf_D21_guard"],
    )
    summary = summary.sort_values(
        ["joint_objective_vs_mainline", "p95_abs_delta_cf_D60_hotspot", "p95_abs_delta_cf_D21_guard", "combo_id"]
    ).reset_index(drop=True)

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
                "input_detail": str(INPUT_DETAIL),
                "input_summary": str(INPUT_SUMMARY),
                "scan": {
                    "beta": BETA,
                    "omega": list(OMEGA_GRID),
                    "m": list(M_GRID),
                    "phase": list(PHASE_GRID),
                    "center_D": list(CENTER_GRID),
                    "width_D": list(WIDTH_GRID),
                    "eta_couple": list(ETA_COUPLE_GRID),
                },
                "notes": (
                    "Counterfactual helical extension on top of the uniform Lorentz hybrid reviewer audit. "
                    "The tested effect is a gated multiplicative damping exp(-omega * gate(D_eff) * notch), "
                    "where notch = (1 - cos(m*(D_eff-center_D) + phase + eta_couple*pi*eta/4))/2. "
                    "This probes whether a non-monotone helical phase can keep the D21-friendly Lorentz signal "
                    "while carving out the D60 hotspot."
                ),
            },
            indent=2,
        )
    )

    for path in [detail_path, summary_path, slices_path, png_path, meta_path]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(summary.head(12).to_string(index=False))
    print(f"\nWrote detail:  {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote slices:  {slices_path}")


if __name__ == "__main__":
    main()
