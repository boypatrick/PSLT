#!/usr/bin/env python3
"""Generate compact figures for the PRD-short PSLT manuscript.

The script intentionally reads existing gated artifacts and produces only
submission-facing summary figures under paper_prd/figures/.  It does not retune
or recompute the release maps.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = ROOT / "paper_prd" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TEX_OUT = ROOT / "paper_prd" / "generated"
TEX_OUT.mkdir(parents=True, exist_ok=True)

TABLE_III_ROW_RE = re.compile(
    r"^\s*(?P<D>\d+(?:\.\d+)?)\s*&\s*"
    r"\$?(?P<E>[-+0-9.]+)\$?\s*&\s*"
    r"(?P<omega>[-+0-9.]+)\s*&\s*"
    r"(?P<S>[-+0-9.]+)\s*&\s*"
    r"\$(?P<mant>[-+0-9.]+)\s*\\times\s*10\^\{(?P<exp>[-+0-9]+)\}\$"
)


def read_first_row(path: Path, key: str | None = None, value: str | None = None) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if key is None:
        return rows[0]
    for row in rows:
        if row.get(key) == value:
            return row
    raise ValueError(f"No row with {key}={value} in {path}")


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def tex_pct(x: float, ndigits: int = 4) -> str:
    return f"{100.0 * x:.{ndigits}f}\\%"


def tex_sci(x: float, sig: int = 6) -> str:
    if x == 0:
        return "0"
    exponent = math.floor(math.log10(abs(x)))
    mantissa = x / (10.0 ** exponent)
    return rf"{mantissa:.{sig}g}\times 10^{{{exponent}}}"


def table_iii_display_residual() -> float:
    table = ROOT / "paper" / "true_unified_table.tex"
    max_resid = 0.0
    for line in table.read_text().splitlines():
        match = TABLE_III_ROW_RE.match(line)
        if not match:
            continue
        S = float(match.group("S"))
        r_val = float(match.group("mant")) * (10.0 ** int(match.group("exp")))
        max_resid = max(max_resid, abs(math.log(r_val) + 2.0 * S))
    if max_resid == 0.0:
        raise RuntimeError(f"No Table III rows parsed from {table}")
    return max_resid


def copy_canonical_maps() -> None:
    copies = {
        ROOT / "output" / "three_generation_phase_diagram.png": OUT / "prd_r3_occupancy_map.png",
    }
    for src, dst in copies.items():
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copyfile(src, dst)


def make_master_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.04, 0.58, 0.18, 0.24, "A1 fixed\ntwo-center\nbackground", "#e6f2ff"),
        (0.28, 0.58, 0.18, 0.24, "Spectral operator\n$[-\\nabla^2+V_{\\rm eff}]\\psi_N=\\omega_N^2\\psi_N$", "#edf7ed"),
        (0.52, 0.58, 0.18, 0.24, "WKB + two lobes\n$S_N$, $r_N=\\eta e^{-2S_N}$", "#fff5df"),
        (0.76, 0.58, 0.18, 0.24, "Rank-2 kinetics\n$\\Gamma_N=\\lambda_+(M_N)$", "#f7e9ff"),
        (0.18, 0.18, 0.20, 0.22, "Effective weight\n$g_N$", "#f0f0f0"),
        (0.42, 0.18, 0.20, 0.22, "Visibility\n$B_N$", "#f0f0f0"),
        (0.66, 0.18, 0.24, 0.22, "Occupancy\n$P_N=W_N/\\sum_K W_K$", "#e9f9f7"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.25,
            edgecolor="#243447",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.4, color="#243447"))

    arrow(0.22, 0.70, 0.28, 0.70)
    arrow(0.46, 0.70, 0.52, 0.70)
    arrow(0.70, 0.70, 0.76, 0.70)
    arrow(0.85, 0.58, 0.80, 0.40)
    arrow(0.38, 0.29, 0.66, 0.29)
    arrow(0.62, 0.29, 0.66, 0.29)
    arrow(0.76, 0.29, 0.76, 0.40)

    ax.text(0.5, 0.94, "PRD-short claim boundary: conditional EFT-level occupancy closure",
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.05, "Not a full EYMH proof of exactly three SM generations; H->mumu is reference-normalized diagnostic only.",
            ha="center", va="center", fontsize=10, color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "prd_master_closure_diagram.png", dpi=220)
    plt.close(fig)


def make_no_fourth_plot() -> None:
    df = pd.read_csv(ROOT / "output" / "highN_decoupling" / "highN_decoupling_single_track_summary.csv")
    D = df["D"].to_numpy(float)
    n = df["n_bound_single_track"].to_numpy(float)
    supports_n4 = df["supports_N4_bound"].to_numpy(int)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(D, n, marker="o", linewidth=2.0, color="#1f5aa6", label="audited bound modes")
    ax.axhline(4, color="#a23b72", linestyle="--", linewidth=1.6, label="fourth-bound-layer threshold")
    ax.fill_between(D, 3.5, 4.5, color="#a23b72", alpha=0.08)
    for x, y in zip(D, n):
        ax.text(x, y + 0.08, f"{int(y)}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel(r"$D_{\rm sep}$")
    ax.set_ylabel("number of negative bound modes")
    ax.set_title("Finite-domain no-fourth-bound-layer certificate")
    ax.set_ylim(0.5, 4.6)
    ax.set_yticks([1, 2, 3, 4])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    ax.text(0.02, 0.04, f"supports_N4_bound = {supports_n4.max()} for all {len(df)} audited D values",
            transform=ax.transAxes, fontsize=10, color="#333333")
    fig.tight_layout()
    fig.savefig(OUT / "prd_no_fourth_bound_certificate.png", dpi=220)
    plt.close(fig)


def make_hmumu_map() -> None:
    map_df = pd.read_csv(ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_map.csv")
    summ = pd.read_csv(ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_summary.csv")
    mumu = summ[summ["channel"] == "mumu"].iloc[0]
    D_vals = np.sort(map_df["D"].unique())
    eta_vals = np.sort(map_df["eta"].unique())
    chi = map_df.pivot(index="eta", columns="D", values="chi2_mumu").loc[eta_vals, D_vals].to_numpy(float)
    mu = map_df.pivot(index="eta", columns="D", values="mu_mumu").loc[eta_vals, D_vals].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    im = ax.imshow(np.clip(chi, 0, 8), origin="lower", aspect="auto",
                   extent=[D_vals.min(), D_vals.max(), eta_vals.min(), eta_vals.max()],
                   cmap="viridis_r", vmin=0, vmax=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"clipped $\chi^2_{\mu\mu}$")
    cs = ax.contour(D_vals, eta_vals, chi, levels=[1, 4], colors=["white", "#ffb000"], linewidths=[1.4, 1.8])
    ax.clabel(cs, inline=True, fmt={1: r"$\chi^2=1$", 4: r"$\chi^2=4$"}, fontsize=9)
    ax.contour(D_vals, eta_vals, mu, levels=[float(mumu["obs_mu"])], colors=["#ff4d4d"], linewidths=1.2)
    ax.plot(float(mumu["best_D"]), float(mumu["best_eta"]), marker="*", markersize=13, color="#ff4d4d", markeredgecolor="black")
    ax.set_xlabel(r"$D_{\rm sep}$")
    ax.set_ylabel(r"$\eta$")
    ax.set_title(r"Reference-normalized $H\to\mu\mu$ diagnostic map")
    ax.text(0.02, 0.04,
            f"obs mu=1.4 +/- 0.4; f(chi2 <= 4)={pct(float(mumu['f_chi2_le_4']))}; best chi2={float(mumu['best_chi2']):.2e}",
            transform=ax.transAxes, fontsize=9.5, color="white",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=4))
    fig.tight_layout()
    fig.savefig(OUT / "prd_hmumu_diagnostic_map.png", dpi=220)
    plt.close(fig)


def make_projector_transport_plot() -> None:
    t3a = read_first_row(ROOT / "output" / "spectral_transport" / "spectral_projector_hf_weighted_bound_summary.csv")
    t3c = read_first_row(ROOT / "output" / "spectral_transport" / "spectral_projector_pullback_variable_box_summary.csv")
    labels = ["rank-one\nprojector drift", "first-three\ncluster drift", "pullback vs G1\neigenvalue drift", "weighted\n||P'_n|| bound"]
    vals = [
        float(t3c["max_rank_one_projector_sin"]),
        float(t3c["max_cluster_sin_theta"]),
        float(t3c["max_abs_pullback_lambda_diff_vs_g1_low3"]),
        float(t3a["max_total_projector_derivative_bound"]),
    ]
    tols = [
        float(t3c["rank_one_drift_tol"]),
        float(t3c["cluster_drift_tol"]),
        np.nan,
        float(t3a["max_total_projector_bound_tol"]),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=["#3066be", "#3aafa9", "#f4a261", "#8d5a97"], alpha=0.88)
    for i, (v, tol) in enumerate(zip(vals, tols)):
        ax.text(i, v * 1.08 + 0.002, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
        if np.isfinite(tol):
            ax.hlines(tol, i - 0.34, i + 0.34, colors="#222222", linestyles="--", linewidth=1.2)
            ax.text(i, tol * 1.08 + 0.002, f"tol {tol:.3g}", ha="center", va="bottom", fontsize=8, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("certificate margin value")
    ax.set_title("Spectral-projector transport margins along D_sep")
    ax.grid(axis="y", which="both", alpha=0.22)
    ax.text(0.02, 0.04, "T3c pullback pass; raw residual Davis-Kahan remains conservative",
            transform=ax.transAxes, fontsize=9.5, color="#333333")
    fig.tight_layout()
    fig.savefig(OUT / "prd_projector_transport_margins.png", dpi=220)
    plt.close(fig)


def write_release_numbers() -> None:
    base = read_first_row(ROOT / "output" / "gn_fp_impact" / "gn_baseline_replacement.csv", "case", "baseline_fp_2d_full")
    mumu = pd.read_csv(ROOT / "output" / "hll_signal_strength" / "hll_signal_strength_summary.csv")
    mumu = mumu[mumu["channel"] == "mumu"].iloc[0].to_dict()
    highn = pd.read_csv(ROOT / "output" / "highN_decoupling" / "highN_decoupling_single_track_summary.csv")
    t3a = read_first_row(ROOT / "output" / "spectral_transport" / "spectral_projector_hf_weighted_bound_summary.csv")
    t3c = read_first_row(ROOT / "output" / "spectral_transport" / "spectral_projector_pullback_variable_box_summary.csv")
    numbers = {
        "R3_gt_0p90": float(base["f_R3_gt_0p90"]),
        "R3_gt_0p95": float(base["f_R3_gt_0p95"]),
        "hmumu_chi2_le_4_from_gn_gate": float(base["f_hmumu_chi2_le_4"]),
        "winner_gt_3": float(base["f_winner_gt_3"]),
        "mean_tail_prob": float(base["mean_tail_prob"]),
        "max_tail_prob": float(base["max_tail_prob"]),
        "hmumu_f_chi2_le_1": float(mumu["f_chi2_le_1"]),
        "hmumu_f_chi2_le_4": float(mumu["f_chi2_le_4"]),
        "hmumu_best_chi2": float(mumu["best_chi2"]),
        "hmumu_best_D": float(mumu["best_D"]),
        "hmumu_best_eta": float(mumu["best_eta"]),
        "highN_n_D": int(len(highn)),
        "highN_supports_N4_max": int(highn["supports_N4_bound"].max()),
        "highN_n_bound_min": int(highn["n_bound_single_track"].min()),
        "highN_n_bound_max": int(highn["n_bound_single_track"].max()),
        "T3a_max_projector_derivative_bound": float(t3a["max_total_projector_derivative_bound"]),
        "T3c_max_rank_one_projector_sin": float(t3c["max_rank_one_projector_sin"]),
        "T3c_max_cluster_sin_theta": float(t3c["max_cluster_sin_theta"]),
        "T3c_max_pullback_lambda_diff_vs_g1_low3": float(t3c["max_abs_pullback_lambda_diff_vs_g1_low3"]),
        "table_iii_max_log_residual_display": table_iii_display_residual(),
    }
    with (OUT / "prd_short_release_numbers.json").open("w") as f:
        json.dump(numbers, f, indent=2, sort_keys=True)
    write_release_tex(numbers)


def write_release_tex(numbers: dict[str, float | int]) -> None:
    macros = [
        r"% Auto-generated by code/generate_prd_short_figures.py; do not edit by hand.",
        rf"\newcommand{{\PRDShortRthreeGTNinety}}{{{tex_pct(float(numbers['R3_gt_0p90']))}}}",
        rf"\newcommand{{\PRDShortRthreeGTNinetyFive}}{{{tex_pct(float(numbers['R3_gt_0p95']))}}}",
        rf"\newcommand{{\PRDShortHmumuChiTwoLeFour}}{{{float(numbers['hmumu_f_chi2_le_4']):.4f}}}",
        rf"\newcommand{{\PRDShortHmumuBestChiTwo}}{{{tex_sci(float(numbers['hmumu_best_chi2']), 6)}}}",
        rf"\newcommand{{\PRDShortHighNDCount}}{{{int(numbers['highN_n_D'])}}}",
        rf"\newcommand{{\PRDShortHighNBoundMin}}{{{int(numbers['highN_n_bound_min'])}}}",
        rf"\newcommand{{\PRDShortHighNBoundMax}}{{{int(numbers['highN_n_bound_max'])}}}",
        rf"\newcommand{{\PRDShortHighNSupportsNFourMax}}{{{int(numbers['highN_supports_N4_max'])}}}",
        rf"\newcommand{{\PRDShortTableIIIResidual}}{{{tex_sci(float(numbers['table_iii_max_log_residual_display']), 6)}}}",
        rf"\newcommand{{\PRDShortTThreeAMaxProjectorDerivative}}{{{float(numbers['T3a_max_projector_derivative_bound']):.17g}}}",
        rf"\newcommand{{\PRDShortTThreeCRankOneDrift}}{{{float(numbers['T3c_max_rank_one_projector_sin']):.17g}}}",
        rf"\newcommand{{\PRDShortTThreeCClusterDrift}}{{{float(numbers['T3c_max_cluster_sin_theta']):.17g}}}",
        rf"\newcommand{{\PRDShortTThreeCLambdaDrift}}{{{tex_sci(float(numbers['T3c_max_pullback_lambda_diff_vs_g1_low3']), 6)}}}",
        "",
    ]
    (TEX_OUT / "prd_short_release_macros.tex").write_text("\n".join(macros))

    table = r"""\begin{table}[htbp]
\centering
\caption{Minimal release-result table for the short manuscript.  The numerical entries are generated from the canonical release artifacts by \texttt{code/generate\_prd\_short\_figures.py}.}
\label{tab:release_results}
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}l l l@{}}
\toprule
\parbox[t]{0.20\textwidth}{\textbf{Observable}} & \parbox[t]{0.34\textwidth}{\textbf{Current release statement}} & \parbox[t]{0.33\textwidth}{\textbf{Short-paper interpretation}} \\
\midrule
\parbox[t]{0.20\textwidth}{First-three occupancy} & \parbox[t]{0.34\textwidth}{\(\Rthree>0.90\) on \(\PRDShortRthreeGTNinety\) of the sampled release grid and \(\Rthree>0.95\) on \(\PRDShortRthreeGTNinetyFive\).} & \parbox[t]{0.33\textwidth}{Evidence for first-three-layer concentration within A1--A5, not a proof of exactly three SM generations.} \\
\parbox[t]{0.20\textwidth}{No-fourth kinetic layer} & \parbox[t]{0.34\textwidth}{On the \(\PRDShortHighNDCount\) audited \(D_{\rm sep}\) values, \(n_{\rm bound}\in[\PRDShortHighNBoundMin,\PRDShortHighNBoundMax]\) and \(\max {\tt supports\_N4\_bound}=\PRDShortHighNSupportsNFourMax\).} & \parbox[t]{0.33\textwidth}{A finite-domain threshold certificate, not a continuum theorem.} \\
\parbox[t]{0.20\textwidth}{Table III WKB consistency} & \parbox[t]{0.34\textwidth}{The canonical gate gives \(\max_D |\log r_1(D)+2S_1(D)|=\PRDShortTableIIIResidual\) for displayed rounded values.} & \parbox[t]{0.33\textwidth}{A numerical consistency gate for the WKB action/tunneling table.} \\
\parbox[t]{0.20\textwidth}{Flavor kernel} & \parbox[t]{0.34\textwidth}{The matched \(H\to\mu\mu\) kernel is row-stochastic and non-idempotent, not a projector.} & \parbox[t]{0.33\textwidth}{Wording and positivity are controlled by the release guard.} \\
\parbox[t]{0.20\textwidth}{\(H\to\mu\mu\) diagnostic} & \parbox[t]{0.34\textwidth}{For \(\mu_{\mu\mu}^{\rm obs}=1.4\pm0.4\), \(f(\chi^2_{\mu\mu}\le4)=\PRDShortHmumuChiTwoLeFour\), with best \(\chi^2=\PRDShortHmumuBestChiTwo\).} & \parbox[t]{0.33\textwidth}{Target-region visualization only; no formal global likelihood is claimed.} \\
\bottomrule
\end{tabular}
\end{table}
"""
    (TEX_OUT / "prd_short_release_results_table.tex").write_text(
        "% Auto-generated by code/generate_prd_short_figures.py; do not edit by hand.\n" + table
    )


def main() -> None:
    copy_canonical_maps()
    make_master_diagram()
    make_no_fourth_plot()
    make_hmumu_map()
    make_projector_transport_plot()
    write_release_numbers()
    print(f"Wrote PRD-short figures to {OUT}")
    print(f"Wrote PRD-short generated TeX to {TEX_OUT}")


if __name__ == "__main__":
    main()
