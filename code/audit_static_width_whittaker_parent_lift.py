#!/usr/bin/env python3
"""
Parent-lift budget audit for the Whittaker-DtN static-width zeros.

S8 certifies that every finite (R, R_tail) Whittaker-DtN root is a local
isolated zero.  This audit asks the next, narrower question: could those local
zeros already be regarded as perturbations of one parent static-width pole?

The audit does not search for a new branch.  It combines:
  - local argument-principle contour data from S8, and
  - optimistic nearest-neighbor branch spans from S7.

For each anchor branch, it records the Rouché residual budget available on the
local contour and the observed finite-parameter energy displacement.  A
parent-side proof would need a tail/conditioning estimate below that budget.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_parent_lift_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_parent_lift_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_parent_lift_budget.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR


DEFAULT_ARGUMENT_DETAIL = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_BRANCH_CANDIDATES = OUTDIR / "static_width_whittaker_branch_isolation_candidates.csv"
DEFAULT_TAG = "static_width_whittaker_parent_lift"


def nearest_argument_rows(branches: pd.DataFrame, argument: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for row in branches.itertuples(index=False):
        D = float(row.D)
        arg_D = argument[argument["D"].astype(float) == D].copy()
        if len(arg_D) == 0:
            continue
        anchor_E = complex(float(row.anchor_E_real), float(row.anchor_E_imag))
        arg_E = arg_D["E0_real"].to_numpy(dtype=float) + 1j * arg_D["E0_imag"].to_numpy(dtype=float)
        idx = int(np.argmin(np.abs(arg_E - anchor_E)))
        arg = arg_D.iloc[idx]
        match_dist = float(abs(arg_E[idx] - anchor_E))

        rho = float(min(float(arg["box_re"]), float(arg["box_im"])))
        boundary_gap = float(arg["boundary_F_abs_min"])
        dF_abs = float(arg["dF_dE_abs_proxy"])
        e_span = float(row.E_abs_span)
        cover_radius = 0.5 * e_span
        square_cover_lower_bound = cover_radius / np.sqrt(2.0)
        linear_contour_budget = dF_abs * rho
        half_span_residual_budget = dF_abs * cover_radius
        full_span_residual_budget = dF_abs * e_span

        local_box_can_cover_branch = cover_radius <= rho
        local_square_can_cover_branch_by_span = square_cover_lower_bound <= rho
        rouche_budget = min(boundary_gap, linear_contour_budget)
        branch_budget_ratio = half_span_residual_budget / max(rouche_budget, 1.0e-300)

        rows.append(
            {
                **row._asdict(),
                "argument_match_dist": match_dist,
                "argument_local_pass": float(arg["local_argument_pass"]),
                "contour_radius_min": rho,
                "boundary_F_abs_min": boundary_gap,
                "dF_dE_abs_proxy": dF_abs,
                "linear_contour_residual_budget": linear_contour_budget,
                "rouche_residual_budget": rouche_budget,
                "branch_cover_radius_required": cover_radius,
                "branch_cover_radius_over_contour": cover_radius / max(rho, 1.0e-300),
                "branch_square_cover_lower_bound": square_cover_lower_bound,
                "branch_square_cover_lower_bound_over_contour": square_cover_lower_bound / max(rho, 1.0e-300),
                "branch_E_span_over_contour": e_span / max(rho, 1.0e-300),
                "half_span_residual_budget": half_span_residual_budget,
                "full_span_residual_budget": full_span_residual_budget,
                "half_span_residual_over_boundary_gap": half_span_residual_budget
                / max(boundary_gap, 1.0e-300),
                "half_span_residual_over_rouche_budget": branch_budget_ratio,
                "local_box_can_cover_branch": float(local_box_can_cover_branch),
                "local_square_can_cover_branch_by_span": float(local_square_can_cover_branch_by_span),
                "parent_lift_status": "passes_local_cover_budget"
                if local_square_can_cover_branch_by_span and branch_budget_ratio <= 1.0
                else "not_lifted_by_current_budget",
            }
        )
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for D, sub in detail.groupby("D", sort=True):
        best = sub.sort_values(
            ["local_box_can_cover_branch", "half_span_residual_over_rouche_budget", "branch_cover_radius_over_contour"],
            ascending=[False, True, True],
        ).iloc[0]
        rows.append(
            {
                "D": float(D),
                "n_anchor_branches": float(len(sub)),
                "n_local_argument_pass": float(sub["argument_local_pass"].sum()),
                "n_local_box_cover_pass": float(sub["local_box_can_cover_branch"].sum()),
                "n_parent_lift_budget_pass": float((sub["parent_lift_status"] == "passes_local_cover_budget").sum()),
                "best_branch_id": float(best["branch_id"]),
                "best_branch_status": str(best["branch_status"]),
                "best_Gamma_candidate": float(best["Gamma_candidate"]),
                "best_E_abs_span": float(best["E_abs_span"]),
                "best_branch_cover_radius_required": float(best["branch_cover_radius_required"]),
                "best_branch_cover_radius_over_contour": float(best["branch_cover_radius_over_contour"]),
                "best_branch_square_cover_lower_bound_over_contour": float(best["branch_square_cover_lower_bound_over_contour"]),
                "best_rouche_residual_budget": float(best["rouche_residual_budget"]),
                "best_half_span_residual_budget": float(best["half_span_residual_budget"]),
                "best_half_span_residual_over_rouche_budget": float(best["half_span_residual_over_rouche_budget"]),
                "min_branch_cover_radius_over_contour": float(sub["branch_cover_radius_over_contour"].min()),
                "min_branch_square_cover_lower_bound_over_contour": float(sub["branch_square_cover_lower_bound_over_contour"].min()),
                "min_half_span_residual_over_rouche_budget": float(sub["half_span_residual_over_rouche_budget"].min()),
                "min_boundary_F_abs_min": float(sub["boundary_F_abs_min"].min()),
                "min_rouche_residual_budget": float(sub["rouche_residual_budget"].min()),
                "parent_lift_status": "not_lifted_current_grid"
                if float((sub["parent_lift_status"] == "passes_local_cover_budget").sum()) == 0
                else "has_budget_passing_branch",
            }
        )

    global_best = detail.sort_values(
        ["local_box_can_cover_branch", "half_span_residual_over_rouche_budget", "branch_cover_radius_over_contour"],
        ascending=[False, True, True],
    ).iloc[0]
    rows.append(
        {
            "D": "global",
            "n_anchor_branches": float(len(detail)),
            "n_local_argument_pass": float(detail["argument_local_pass"].sum()),
            "n_local_box_cover_pass": float(detail["local_box_can_cover_branch"].sum()),
            "n_parent_lift_budget_pass": float((detail["parent_lift_status"] == "passes_local_cover_budget").sum()),
            "best_branch_id": float(global_best["branch_id"]),
            "best_branch_status": str(global_best["branch_status"]),
            "best_Gamma_candidate": float(global_best["Gamma_candidate"]),
            "best_E_abs_span": float(global_best["E_abs_span"]),
            "best_branch_cover_radius_required": float(global_best["branch_cover_radius_required"]),
            "best_branch_cover_radius_over_contour": float(global_best["branch_cover_radius_over_contour"]),
            "best_branch_square_cover_lower_bound_over_contour": float(global_best["branch_square_cover_lower_bound_over_contour"]),
            "best_rouche_residual_budget": float(global_best["rouche_residual_budget"]),
            "best_half_span_residual_budget": float(global_best["half_span_residual_budget"]),
            "best_half_span_residual_over_rouche_budget": float(global_best["half_span_residual_over_rouche_budget"]),
            "min_branch_cover_radius_over_contour": float(detail["branch_cover_radius_over_contour"].min()),
            "min_branch_square_cover_lower_bound_over_contour": float(detail["branch_square_cover_lower_bound_over_contour"].min()),
            "min_half_span_residual_over_rouche_budget": float(detail["half_span_residual_over_rouche_budget"].min()),
            "min_boundary_F_abs_min": float(detail["boundary_F_abs_min"].min()),
            "min_rouche_residual_budget": float(detail["rouche_residual_budget"].min()),
            "parent_lift_status": "not_lifted_current_grid"
            if float((detail["parent_lift_status"] == "passes_local_cover_budget").sum()) == 0
            else "has_budget_passing_branch",
        }
    )
    return pd.DataFrame(rows)


def plot_budget(detail: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    colors = {6.0: "#1f77b4", 12.0: "#d95f02", 18.0: "#2ca02c"}

    ax = axes[0]
    for D, sub in detail.groupby("D", sort=True):
        ax.scatter(
            sub["branch_cover_radius_over_contour"],
            sub["half_span_residual_over_rouche_budget"],
            s=42,
            alpha=0.78,
            label=f"D={D:g}",
            color=colors.get(float(D), None),
        )
    ax.axvline(1.0, color="black", lw=1.0, ls="--")
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("required cover radius / local contour radius")
    ax.set_ylabel("half-span residual / Rouché budget")
    ax.set_title("Parent-lift budget test")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    grouped = detail.groupby("D", sort=True)["branch_cover_radius_over_contour"].min()
    ax.bar([str(int(k)) for k in grouped.index], grouped.to_numpy(dtype=float), color="#586e75")
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("D")
    ax.set_ylabel("best cover-radius ratio")
    ax.set_title("Best audited branch still misses local cover budget")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--argument-detail", type=Path, default=DEFAULT_ARGUMENT_DETAIL)
    ap.add_argument("--branch-candidates", type=Path, default=DEFAULT_BRANCH_CANDIDATES)
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    argument = pd.read_csv(args.argument_detail)
    branches = pd.read_csv(args.branch_candidates)
    detail = nearest_argument_rows(branches=branches, argument=argument)
    if len(detail) == 0:
        raise RuntimeError("No parent-lift rows produced.")
    summary = summarize(detail)

    detail_path = OUTDIR / f"{args.out_tag}_detail.csv"
    summary_path = OUTDIR / f"{args.out_tag}_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))

    if not args.skip_plot:
        plot_path = OUTDIR / f"{args.out_tag}_budget.png"
        plot_budget(detail, plot_path)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
