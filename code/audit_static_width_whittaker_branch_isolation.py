#!/usr/bin/env python3
"""
Whittaker-DtN branch-isolation audit for the static-width gate.

Proposition S6 audits the full Coulomb/Whittaker outgoing DtN backend, but the
selected roots still drift under the finite interface R and the matching radius
R_tail.  This follow-up keeps the same backend and asks a sharper branch
question:

    If a resolved exterior-stable pole is already present among the computed
    Whittaker-DtN roots, can an optimistic nearest-neighbor continuation find it?

For every root in the Whittaker-DtN detail table, the audit forms the branch

    B_a(R_tail, R, parity) = argmin_z |z - E_a|,

where z ranges over the resolved roots for the same D and parameter tuple.  The
branch is deliberately favorable to promotion: it does not require preserving
the originally selected seed label.  A resolved stable pole in this candidate
set should therefore pass at least one anchor branch.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_branch_isolation_candidates.csv
  - output/cap_resonance_1d/static_width_whittaker_branch_isolation_selected.csv
  - output/cap_resonance_1d/static_width_whittaker_branch_isolation_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_branch_isolation_roots.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR, relative_span


@dataclass(frozen=True)
class BranchIsolationConfig:
    detail_csv: Path = OUTDIR / "static_width_whittaker_dtn_dps50_detail.csv"
    out_tag: str = "static_width_whittaker_branch_isolation"
    r_gamma_rel_span_max: float = 0.15
    tail_gamma_rel_span_max: float = 0.05
    r120_over_r80_min: float = 0.90
    r120_over_r80_max: float = 1.10
    parity_gamma_rel_split_max: float = 1.0e-5
    root_residual_abs_max: float = 1.0e-6


def complex_values(df: pd.DataFrame) -> np.ndarray:
    return df["E_dtn_real"].to_numpy(dtype=float) + 1j * df["E_dtn_imag"].to_numpy(dtype=float)


def complex_pairwise_span(values: Iterable[complex]) -> float:
    vals = np.asarray(list(values), dtype=complex)
    vals = vals[np.isfinite(vals.real) & np.isfinite(vals.imag)]
    if len(vals) < 2:
        return np.nan
    return float(np.max(np.abs(vals[:, None] - vals[None, :])))


def parity_split_metrics(branch: pd.DataFrame) -> tuple[float, float]:
    even = branch[branch["parity"] == "even"].set_index(["R_tail", "R"])
    odd = branch[branch["parity"] == "odd"].set_index(["R_tail", "R"])
    common = sorted(set(even.index).intersection(set(odd.index)))
    gamma_rel: list[float] = []
    e_abs: list[float] = []
    for key in common:
        ge = float(even.loc[key, "Gamma_dtn"])
        go = float(odd.loc[key, "Gamma_dtn"])
        ee = complex(float(even.loc[key, "E_dtn_real"]), float(even.loc[key, "E_dtn_imag"]))
        eo = complex(float(odd.loc[key, "E_dtn_real"]), float(odd.loc[key, "E_dtn_imag"]))
        gamma_rel.append(abs(ge - go) / max(0.5 * (abs(ge) + abs(go)), 1.0e-300))
        e_abs.append(abs(ee - eo))
    return (
        float(max(gamma_rel)) if gamma_rel else np.nan,
        float(max(e_abs)) if e_abs else np.nan,
    )


def summarize_branch(branch: pd.DataFrame, anchor: pd.Series, branch_id: int, cfg: BranchIsolationConfig) -> dict[str, float | str]:
    by_tail_R = branch.groupby(["R_tail", "R"], sort=True)["Gamma_dtn"].mean()
    r_spans: list[float] = []
    r_ratios: list[float] = []
    for _, tail_sub in by_tail_R.groupby(level=0, sort=True):
        vals = tail_sub.droplevel(0)
        r_spans.append(relative_span(vals.to_numpy(dtype=float)))
        if 80.0 in vals.index and 120.0 in vals.index and float(vals.loc[80.0]) != 0.0:
            r_ratios.append(float(vals.loc[120.0] / vals.loc[80.0]))

    tail_spans: list[float] = []
    for _, R_sub in by_tail_R.groupby(level=1, sort=True):
        vals = R_sub.droplevel(1)
        tail_spans.append(relative_span(vals.to_numpy(dtype=float)))

    parity_gamma_rel_max, parity_E_abs_max = parity_split_metrics(branch)
    residual_abs_max = float(branch["residual_abs"].max())
    r_span_max = float(max(r_spans)) if r_spans else np.nan
    tail_span_max = float(max(tail_spans)) if tail_spans else np.nan
    r_ratio_min = float(min(r_ratios)) if r_ratios else np.nan
    r_ratio_max = float(max(r_ratios)) if r_ratios else np.nan

    reasons: list[str] = []
    if residual_abs_max > cfg.root_residual_abs_max:
        reasons.append("root_residual_large")
    if parity_gamma_rel_max > cfg.parity_gamma_rel_split_max:
        reasons.append("parity_split_large")
    if r_span_max > cfg.r_gamma_rel_span_max:
        reasons.append("R_width_not_stationary")
    if tail_span_max > cfg.tail_gamma_rel_span_max:
        reasons.append("R_tail_not_stationary")
    if not (
        np.isfinite(r_ratio_min)
        and np.isfinite(r_ratio_max)
        and cfg.r120_over_r80_min <= r_ratio_min
        and r_ratio_max <= cfg.r120_over_r80_max
    ):
        reasons.append("R120_R80_ratio_outside_gate")

    normalized = [
        r_span_max / max(cfg.r_gamma_rel_span_max, 1.0e-300),
        tail_span_max / max(cfg.tail_gamma_rel_span_max, 1.0e-300),
        parity_gamma_rel_max / max(cfg.parity_gamma_rel_split_max, 1.0e-300),
        residual_abs_max / max(cfg.root_residual_abs_max, 1.0e-300),
    ]
    if np.isfinite(r_ratio_min) and r_ratio_min < cfg.r120_over_r80_min:
        normalized.append((cfg.r120_over_r80_min - r_ratio_min) / max(1.0 - cfg.r120_over_r80_min, 1.0e-300))
    if np.isfinite(r_ratio_max) and r_ratio_max > cfg.r120_over_r80_max:
        normalized.append((r_ratio_max - cfg.r120_over_r80_max) / max(cfg.r120_over_r80_max - 1.0, 1.0e-300))

    energies = complex_values(branch)
    pass_gate = len(reasons) == 0
    return {
        "D": float(anchor["D"]),
        "branch_id": float(branch_id),
        "anchor_R": float(anchor["R"]),
        "anchor_R_tail": float(anchor["R_tail"]),
        "anchor_parity": str(anchor["parity"]),
        "anchor_E_real": float(anchor["E_dtn_real"]),
        "anchor_E_imag": float(anchor["E_dtn_imag"]),
        "anchor_Gamma": float(anchor["Gamma_dtn"]),
        "n_cases": float(len(branch)),
        "Gamma_min": float(branch["Gamma_dtn"].min()),
        "Gamma_max": float(branch["Gamma_dtn"].max()),
        "Gamma_candidate": float(np.median(by_tail_R.to_numpy(dtype=float))),
        "Gamma_R_rel_span_max": r_span_max,
        "Gamma_Rtail_rel_span_max": tail_span_max,
        "Gamma_R120_over_R80_min": r_ratio_min,
        "Gamma_R120_over_R80_max": r_ratio_max,
        "parity_Gamma_rel_split_max": parity_gamma_rel_max,
        "parity_E_abs_split_max": parity_E_abs_max,
        "E_abs_span": complex_pairwise_span(energies),
        "root_residual_abs_max": residual_abs_max,
        "branch_pick_dist_max": float(branch["branch_pick_dist"].max()),
        "branch_pick_dist_median": float(branch["branch_pick_dist"].median()),
        "branch_stability_score": float(np.nanmax(normalized)),
        "branch_pass": float(pass_gate),
        "branch_status": "whittaker_stable_branch" if pass_gate else "not_promoted",
        "fail_reasons": "pass" if pass_gate else ";".join(reasons),
    }


def build_branches(detail: pd.DataFrame, cfg: BranchIsolationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    branch_rows: list[dict[str, float | str]] = []
    selected_rows: list[dict[str, float | str]] = []

    for D, Ddf in detail.groupby("D", sort=True):
        groups = {key: sub.copy() for key, sub in Ddf.groupby(["R_tail", "R", "parity"], sort=True)}
        for branch_id, (_, anchor) in enumerate(Ddf.reset_index(drop=True).iterrows()):
            anchor_E = complex(float(anchor["E_dtn_real"]), float(anchor["E_dtn_imag"]))
            selected: list[pd.Series] = []
            for key, sub in groups.items():
                vals = complex_values(sub)
                idx = int(np.argmin(np.abs(vals - anchor_E)))
                row = sub.iloc[idx].copy()
                row["branch_id"] = float(branch_id)
                row["anchor_E_real"] = float(anchor_E.real)
                row["anchor_E_imag"] = float(anchor_E.imag)
                row["branch_pick_dist"] = float(abs(vals[idx] - anchor_E))
                row["case_key"] = f"Rtail={key[0]:g};R={key[1]:g};parity={key[2]}"
                selected.append(row)
            branch = pd.DataFrame(selected)
            branch_rows.append(summarize_branch(branch=branch, anchor=anchor, branch_id=branch_id, cfg=cfg))
            selected_rows.extend(row.to_dict() for _, row in branch.iterrows())

    return pd.DataFrame(branch_rows), pd.DataFrame(selected_rows)


def summarize_all(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for D, sub in candidates.groupby("D", sort=True):
        best = sub.sort_values(["branch_pass", "branch_stability_score"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "D": float(D),
                "n_anchor_branches": float(len(sub)),
                "n_passing_branches": float(sub["branch_pass"].sum()),
                "best_branch_id": float(best["branch_id"]),
                "best_branch_status": str(best["branch_status"]),
                "best_branch_stability_score": float(best["branch_stability_score"]),
                "best_Gamma_candidate": float(best["Gamma_candidate"]),
                "best_Gamma_R_rel_span_max": float(best["Gamma_R_rel_span_max"]),
                "best_Gamma_Rtail_rel_span_max": float(best["Gamma_Rtail_rel_span_max"]),
                "best_Gamma_R120_over_R80_min": float(best["Gamma_R120_over_R80_min"]),
                "best_Gamma_R120_over_R80_max": float(best["Gamma_R120_over_R80_max"]),
                "best_parity_Gamma_rel_split_max": float(best["parity_Gamma_rel_split_max"]),
                "best_E_abs_span": float(best["E_abs_span"]),
                "best_branch_pick_dist_max": float(best["branch_pick_dist_max"]),
                "best_fail_reasons": str(best["fail_reasons"]),
            }
        )

    global_row = {
        "D": "global",
        "n_anchor_branches": float(len(candidates)),
        "n_passing_branches": float(candidates["branch_pass"].sum()),
        "best_branch_id": float(candidates.sort_values("branch_stability_score").iloc[0]["branch_id"]),
        "best_branch_status": "whittaker_stable_branch_global"
        if float(candidates["branch_pass"].sum()) > 0
        else "not_promoted_global",
        "best_branch_stability_score": float(candidates["branch_stability_score"].min()),
        "best_Gamma_candidate": float(candidates["Gamma_candidate"].median()),
        "best_Gamma_R_rel_span_max": float(candidates["Gamma_R_rel_span_max"].min()),
        "best_Gamma_Rtail_rel_span_max": float(candidates["Gamma_Rtail_rel_span_max"].min()),
        "best_Gamma_R120_over_R80_min": float(candidates["Gamma_R120_over_R80_min"].max()),
        "best_Gamma_R120_over_R80_max": float(candidates["Gamma_R120_over_R80_max"].min()),
        "best_parity_Gamma_rel_split_max": float(candidates["parity_Gamma_rel_split_max"].min()),
        "best_E_abs_span": float(candidates["E_abs_span"].min()),
        "best_branch_pick_dist_max": float(candidates["branch_pick_dist_max"].min()),
        "best_fail_reasons": ";".join(sorted(set(candidates["fail_reasons"].astype(str)))),
    }
    rows.append(global_row)
    return pd.DataFrame(rows)


def plot_roots(detail: pd.DataFrame, candidates: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(sorted(detail["D"].unique())), figsize=(13, 3.8), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    colors = {400.0: "#c45a2a", 800.0: "#1d6fa5"}
    markers = {"even": "o", "odd": "s"}
    for ax, (D, sub) in zip(axes, detail.groupby("D", sort=True)):
        for row in sub.itertuples():
            ax.scatter(
                row.E_dtn_real,
                row.E_dtn_imag,
                s=34 + 0.15 * row.R,
                c=colors.get(float(row.R_tail), "#555555"),
                marker=markers.get(str(row.parity), "o"),
                edgecolors="black",
                linewidths=0.35,
                alpha=0.82,
            )
            ax.text(row.E_dtn_real, row.E_dtn_imag, f"{int(row.R)}", fontsize=6, ha="center", va="center")
        best = candidates[candidates["D"] == float(D)].sort_values("branch_stability_score").iloc[0]
        ax.scatter(
            best["anchor_E_real"],
            best["anchor_E_imag"],
            s=145,
            facecolors="none",
            edgecolors="#0b0b0b",
            linewidths=1.4,
            label="best anchor",
        )
        ax.axhline(0.0, color="0.75", lw=0.8)
        ax.set_title(f"D={D:g}, pass=0/{len(candidates[candidates['D'] == float(D)])}")
        ax.set_xlabel(r"$\Re E$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$\Im E$")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[400.0], markeredgecolor="black", label=r"$R_C=400$", markersize=7),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[800.0], markeredgecolor="black", label=r"$R_C=800$", markersize=7),
        plt.Line2D([0], [0], marker="o", color="black", linestyle="None", label="even", markersize=6),
        plt.Line2D([0], [0], marker="s", color="black", linestyle="None", label="odd", markersize=6),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Whittaker-DtN branch-isolation audit.")
    ap.add_argument("--detail-csv", default=str(BranchIsolationConfig.detail_csv), help="Whittaker-DtN detail CSV.")
    ap.add_argument("--out-tag", default=BranchIsolationConfig.out_tag, help="Output filename stem.")
    ap.add_argument("--skip-plot", action="store_true", help="Do not write the root-cloud plot.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BranchIsolationConfig(detail_csv=Path(args.detail_csv), out_tag=str(args.out_tag))
    if not cfg.detail_csv.exists():
        raise FileNotFoundError(f"Missing Whittaker detail CSV: {cfg.detail_csv}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(cfg.detail_csv)
    required = {"D", "R", "R_tail", "parity", "E_dtn_real", "E_dtn_imag", "Gamma_dtn", "residual_abs"}
    missing = sorted(required.difference(detail.columns))
    if missing:
        raise ValueError(f"Missing required detail columns: {missing}")

    candidates, selected = build_branches(detail=detail, cfg=cfg)
    summary = summarize_all(candidates)

    candidate_out = OUTDIR / f"{cfg.out_tag}_candidates.csv"
    selected_out = OUTDIR / f"{cfg.out_tag}_selected.csv"
    summary_out = OUTDIR / f"{cfg.out_tag}_summary.csv"
    candidates.to_csv(candidate_out, index=False)
    selected.to_csv(selected_out, index=False)
    summary.to_csv(summary_out, index=False)

    plot_out = OUTDIR / f"{cfg.out_tag}_roots.png"
    if not args.skip_plot:
        plot_roots(detail=detail, candidates=candidates, out_path=plot_out)

    print(f"Wrote {candidate_out}")
    print(f"Wrote {selected_out}")
    print(f"Wrote {summary_out}")
    if not args.skip_plot:
        print(f"Wrote {plot_out}")
    cols = [
        "D",
        "n_anchor_branches",
        "n_passing_branches",
        "best_branch_stability_score",
        "best_Gamma_R_rel_span_max",
        "best_Gamma_Rtail_rel_span_max",
        "best_Gamma_R120_over_R80_min",
        "best_Gamma_R120_over_R80_max",
        "best_parity_Gamma_rel_split_max",
        "best_fail_reasons",
    ]
    print("[Whittaker-DtN branch-isolation summary]")
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
