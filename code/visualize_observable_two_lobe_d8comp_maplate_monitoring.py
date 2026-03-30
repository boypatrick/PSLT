#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "kinetic_action_chain"
PAPER_DIR = ROOT / "paper"

SUMMARY_CSV = OUTDIR / "observable_two_lobe_d8comp_maplate_d60_monitoring_audit_summary.csv"
SLICES_CSV = OUTDIR / "observable_two_lobe_d8comp_maplate_d60_monitoring_audit_slices.csv"
DETAIL_CSV = OUTDIR / "observable_two_lobe_d8comp_maplate_d60_monitoring_audit_detail.csv"

DIST_PNG = OUTDIR / "observable_two_lobe_d8comp_maplate_monitoring_distribution_view.png"
FORMULA_PNG = OUTDIR / "observable_two_lobe_d8comp_maplate_formula_graphics.png"
CURVES_CSV = OUTDIR / "observable_two_lobe_d8comp_maplate_formula_curves.csv"

CASE_ORDER = ["D60E21_monitor", "D60E41_monitor", "D80E41_monitor", "D100E41_monitor"]


def _parse_targets(raw: str) -> list[float]:
    vals: list[float] = []
    for tok in str(raw).split("|"):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals


def _case_meta(summary: pd.DataFrame) -> dict[str, dict[str, object]]:
    meta: dict[str, dict[str, object]] = {}
    for case_name in CASE_ORDER:
        grp = summary.loc[summary["case"] == case_name]
        if grp.empty:
            continue
        row = grp.iloc[0]
        meta[case_name] = {
            "reviewer_ds": _parse_targets(row["reviewer_grid_targets"]),
            "drift_grid_D": float(row["drift_grid_D"]),
        }
    return meta


def _subset_mask(detail: pd.DataFrame, reviewer_ds: list[float], drift_d: float, subset: str) -> np.ndarray:
    dvals = detail["D"].to_numpy(dtype=float)
    reviewer_mask = np.zeros(len(detail), dtype=bool)
    for d in reviewer_ds:
        reviewer_mask |= np.isclose(dvals, d, atol=1.0e-9)
    drift_mask = np.isclose(dvals, float(drift_d), atol=1.0e-9)
    if subset == "all_points":
        return np.ones(len(detail), dtype=bool)
    if subset == "reviewer_targets":
        return reviewer_mask
    if subset == "nonreviewer_complement":
        return ~reviewer_mask
    if subset == "drift_strip":
        return drift_mask
    raise ValueError(subset)


def _draw_box_panel(ax: plt.Axes, detail: pd.DataFrame, meta: dict[str, dict[str, object]], subset: str, title: str) -> None:
    data = []
    labels = []
    for case_name in CASE_ORDER:
        if case_name not in meta:
            continue
        case_df = detail.loc[detail["case"] == case_name].copy()
        mask = _subset_mask(case_df, meta[case_name]["reviewer_ds"], meta[case_name]["drift_grid_D"], subset)
        vals = case_df.loc[mask, "abs_improvement_vs_parent"].to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(case_name.replace("_monitor", ""))
    ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True, medianprops={"color": "black"})
    for patch, color in zip(ax.artists, ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.45)
    ax.set_title(title)
    ax.set_ylabel("candidate-parent improvement")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.2)


def _draw_heatmap(ax: plt.Axes, detail: pd.DataFrame, case_name: str) -> None:
    case_df = detail.loc[detail["case"] == case_name].copy()
    pivot = case_df.pivot(index="eta", columns="D", values="abs_improvement_vs_parent").sort_index()
    vmax = float(np.nanpercentile(np.abs(pivot.to_numpy(dtype=float)), 99.0))
    vmax = max(vmax, 1.0e-6)
    im = ax.imshow(
        pivot.to_numpy(dtype=float),
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        extent=[float(pivot.columns.min()), float(pivot.columns.max()), float(pivot.index.min()), float(pivot.index.max())],
    )
    ax.set_title(case_name.replace("_monitor", ""))
    ax.set_xlabel("D")
    ax.set_ylabel("eta")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="candidate-parent improvement")


def plot_distribution_view(summary: pd.DataFrame, slices: pd.DataFrame, detail: pd.DataFrame) -> None:
    meta = _case_meta(summary)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    _draw_box_panel(axes[0, 0], detail, meta, "all_points", "All-Points Distribution")
    _draw_box_panel(axes[0, 1], detail, meta, "nonreviewer_complement", "Non-Reviewer Complement")
    _draw_box_panel(axes[0, 2], detail, meta, "drift_strip", "Drift-Strip Distribution")

    ax = axes[1, 0]
    palette = {
        "D60E21_monitor": "#4e79a7",
        "D60E41_monitor": "#59a14f",
        "D80E41_monitor": "#f28e2b",
        "D100E41_monitor": "#e15759",
    }
    for case_name in CASE_ORDER:
        grp = slices.loc[slices["case"] == case_name].sort_values("D")
        if grp.empty:
            continue
        ax.plot(grp["D"], grp["slice_mean_abs_improvement_vs_parent"], marker="o", lw=1.5, ms=3.0, color=palette[case_name], label=case_name.replace("_monitor", ""))
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.45)
    ax.set_title("Slice Mean Improvement vs D")
    ax.set_xlabel("D")
    ax.set_ylabel("candidate-parent improvement")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    _draw_heatmap(axes[1, 1], detail, "D80E41_monitor")
    _draw_heatmap(axes[1, 2], detail, "D100E41_monitor")

    fig.suptitle("D8comp Monitoring Distribution View", fontsize=16, fontweight="bold")
    fig.savefig(DIST_PNG, dpi=180)
    plt.close(fig)


def _two_lobe_curves() -> pd.DataFrame:
    d = np.linspace(4.0, 20.0, 2400)
    beta = 0.4
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    d_eff = d / gamma
    omega = 1.3
    m = 14.0
    phase = 0.4353981633974483
    center = 5.8
    width = 0.07
    boost = 0.4
    boost_center = 5.86
    boost_width = 0.05

    z = (d_eff - center) / width
    gate = 1.0 / (1.0 + np.exp(-z))
    theta = m * (d_eff - center) + phase
    notch = 0.5 * (1.0 - np.cos(theta))
    suppress = np.exp(-omega * gate * notch)
    boost_gaussian = np.exp(-0.5 * ((d_eff - boost_center) / boost_width) ** 2)
    compensate = np.exp(boost * boost_gaussian)
    two_lobe_factor = suppress * compensate

    res_d8 = np.clip(1.0 - 0.70 * np.exp(-0.5 * ((d - 7.95) / 0.10) ** 2), 0.0, 1.0)
    res_d72 = np.clip(1.0 - 1.00 * np.exp(-0.5 * ((d - 7.20) / 0.02) ** 2), 0.0, 1.0)
    res_d96 = np.clip(1.0 - 0.30 * np.exp(-0.5 * ((d - 9.60) / 0.02) ** 2), 0.0, 1.0)
    residual_total = res_d8 * res_d72 * res_d96

    return pd.DataFrame(
        {
            "D": d,
            "D_eff": d_eff,
            "gate": gate,
            "theta": theta,
            "notch": notch,
            "suppress": suppress,
            "boost_gaussian": boost_gaussian,
            "compensate": compensate,
            "two_lobe_factor": two_lobe_factor,
            "residual_d8": res_d8,
            "residual_d72": res_d72,
            "residual_d96": res_d96,
            "residual_total": residual_total,
        }
    )


def plot_formula_view(summary: pd.DataFrame) -> None:
    curves = _two_lobe_curves()
    curves.to_csv(CURVES_CSV, index=False)

    d80 = float(summary.loc[(summary["case"] == "D80E41_monitor") & (summary["subset"] == "drift_strip"), "drift_grid_D"].iloc[0])
    d100 = float(summary.loc[(summary["case"] == "D100E41_monitor") & (summary["subset"] == "drift_strip"), "drift_grid_D"].iloc[0])
    d60 = float(summary.loc[(summary["case"] == "D60E41_monitor") & (summary["subset"] == "drift_strip"), "drift_grid_D"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(curves["D"], curves["D_eff"], color="#4e79a7", label=r"$D_{\mathrm{eff}}=D/\gamma$")
    ax2 = ax.twinx()
    ax2.plot(curves["D"], curves["gate"], color="#e15759", label="gate")
    ax.set_title("Lorentz Reparameterization")
    ax.set_xlabel("D")
    ax.set_ylabel("D_eff")
    ax2.set_ylabel("gate")
    ax.grid(True, alpha=0.2)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], frameon=False, loc="upper left")

    ax = axes[0, 1]
    ax.plot(curves["D"], curves["notch"], label="notch", color="#bab0ab")
    ax.plot(curves["D"], curves["suppress"], label="suppress", color="#e15759")
    ax.plot(curves["D"], curves["compensate"], label="compensate", color="#59a14f")
    ax.plot(curves["D"], curves["two_lobe_factor"], label="two-lobe factor", color="#4e79a7", lw=2.0)
    ax.set_title("Two-Lobe Kernel")
    ax.set_xlabel("D")
    ax.set_ylabel("weight")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(curves["D"], curves["residual_d8"], label="D8 strip", color="#4e79a7")
    ax.plot(curves["D"], curves["residual_d72"], label="D7.2 strip", color="#f28e2b")
    ax.plot(curves["D"], curves["residual_d96"], label="D9.6 strip", color="#59a14f")
    ax.plot(curves["D"], curves["residual_total"], label="combined residual", color="#e15759", lw=2.0)
    for x, label in [(7.20, "7.2"), (7.95, "7.95"), (9.60, "9.6"), (d60, "D60 drift"), (d80, "D80 drift"), (d100, "D100 drift")]:
        ax.axvline(x, color="black", lw=0.8, alpha=0.2)
        ax.text(x, 1.015, label, rotation=90, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Map-Layer Repair Kernels")
    ax.set_xlabel("D")
    ax.set_ylabel("residual fraction")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.axis("off")
    formula_text = "\n".join(
        [
            r"$D_{\mathrm{eff}} = D / \gamma,\quad \gamma=(1-\beta^2)^{-1/2},\ \beta=0.4$",
            r"$\mathrm{notch}(D)=\frac{1-\cos(m(D_{\mathrm{eff}}-D_c)+\phi)}{2}$",
            r"$F_{\mathrm{2lobe}}(D)=e^{-\omega\,g(D_{\mathrm{eff}})\,\mathrm{notch}(D)}\;"
            r"e^{\,b\exp[-(D_{\mathrm{eff}}-D_b)^2/(2w_b^2)]}$",
            "",
            r"$R_{\mathrm{map}}(D)=\prod_{k\in\{7.2,7.95,9.6\}}"
            r"\max\!\left(1-p_k e^{-(D-c_k)^2/(2w_k^2)},\,0\right)$",
            r"$\mu_{\mathrm{cf}}(D,\eta)=\mu_{\mathrm{base}}(D,\eta)+"
            r"(\mu_{\mathrm{runtime}}-\mu_{\mathrm{base}})\,R_{\mathrm{map}}(D)$",
            "",
            "What the plots are showing:",
            "1. The two-lobe factor is a D-only suppression/compensation kernel in Lorentz-reparameterized D_eff.",
            "2. The late map-layer repair is another D-only residual blend, with narrow strips at 7.2 and 9.6 plus a broader D8 repair.",
            "3. The D≈7.25 drift band sits between the 7.2 strip and the D8 repair. At D80/D100 it is no longer negligible.",
        ]
    )
    ax.text(0.0, 1.0, formula_text, ha="left", va="top", fontsize=11, family="serif")

    fig.suptitle("D8comp Formula Graphics", fontsize=16, fontweight="bold")
    fig.savefig(FORMULA_PNG, dpi=180)
    plt.close(fig)


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    slices = pd.read_csv(SLICES_CSV)
    detail = pd.read_csv(DETAIL_CSV)

    plot_distribution_view(summary, slices, detail)
    plot_formula_view(summary)

    for path in [DIST_PNG, FORMULA_PNG, CURVES_CSV]:
        (PAPER_DIR / path.name).write_bytes(path.read_bytes())

    print(f"Wrote distribution figure: {DIST_PNG}")
    print(f"Wrote formula figure:      {FORMULA_PNG}")
    print(f"Wrote formula curves:      {CURVES_CSV}")


if __name__ == "__main__":
    main()
