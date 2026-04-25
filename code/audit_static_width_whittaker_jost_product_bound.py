#!/usr/bin/env python3
"""
Analytic Jost-product / projective pole-exclusion budget audit.

This is the theorem-facing follow-up to S12.  It does not compute new roots,
does not introduce a new tail backend, and does not change the static-width
promotion object.  It reads the already-audited finite-amplitude projective
transfer rows and checks the parent-side sufficient inequality

    |u_a(R) u_b(R)| > |a-b| / B_p^(m),

where a and b are the two dyadic outgoing slopes at R0 and B_p^(m) is the
local Rouche m-budget inherited from the S8 contour.

The same inequality is also written as projective pole exclusion.  If

    u_s(R) = A + B s = B(s-p),      p = -A/B,

then

    |u_a(R)u_b(R)| = |B|^2 |a-p| |b-p|.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_jost_product_bound_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_jost_product_bound_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_jost_product_bound_budget.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR


DEFAULT_PROJECTIVE = OUTDIR / "static_width_whittaker_projective_transfer_detail.csv"
DEFAULT_TAG = "static_width_whittaker_jost_product_bound"


def complex_columns(df: pd.DataFrame, real_col: str, imag_col: str) -> np.ndarray:
    return df[real_col].to_numpy(dtype=float) + 1j * df[imag_col].to_numpy(dtype=float)


def build_detail(projective: pd.DataFrame, pair: str, slack: float) -> pd.DataFrame:
    df = projective[projective["transfer_pair"] == pair].copy()
    if df.empty:
        raise ValueError(f"No rows found for transfer_pair={pair!r}")

    a = complex_columns(df, "a_start_real", "a_start_imag")
    b = complex_columns(df, "b_start_real", "b_start_imag")
    pole = complex_columns(df, "jost_pole_initial_real", "jost_pole_initial_imag")

    da = np.abs(a - pole)
    db = np.abs(b - pole)
    initial_delta = df["initial_delta_at_R_start"].to_numpy(dtype=float)
    budget = df["m_rouche_budget"].to_numpy(dtype=float)
    product = df["jost_denom_product_abs"].to_numpy(dtype=float)
    denom_a = df["jost_denom_a_abs"].to_numpy(dtype=float)
    denom_b = df["jost_denom_b_abs"].to_numpy(dtype=float)

    required_product = initial_delta / np.maximum(budget, 1.0e-300)
    product_margin = product / np.maximum(required_product, 1.0e-300)

    transfer_ratio = df["transfer_delta_over_budget"].to_numpy(dtype=float)
    exact_ratio = df["exact_projective_delta_over_budget"].to_numpy(dtype=float)
    formula_abs_error_ratio = df["exact_projective_formula_abs_error_over_budget"].to_numpy(dtype=float)
    transfer_upper_ratio = np.maximum(transfer_ratio, exact_ratio) + formula_abs_error_ratio
    product_lower_from_transfer = required_product / np.maximum(transfer_upper_ratio, 1.0e-300)
    product_lower_from_transfer *= max(0.0, 1.0 - slack)
    product_lower_margin = product_lower_from_transfer / np.maximum(required_product, 1.0e-300)

    B_abs_from_a = denom_a / np.maximum(da, 1.0e-300)
    B_abs_from_b = denom_b / np.maximum(db, 1.0e-300)
    B_abs_floor = np.minimum(B_abs_from_a, B_abs_from_b) * max(0.0, 1.0 - slack)
    pole_product_lower = (B_abs_floor**2) * da * db * max(0.0, 1.0 - slack)
    pole_product_margin = pole_product_lower / np.maximum(required_product, 1.0e-300)
    B_abs_rel_mismatch = np.abs(B_abs_from_a - B_abs_from_b) / np.maximum(
        np.maximum(B_abs_from_a, B_abs_from_b),
        1.0e-300,
    )
    pole_identity_rel_error = np.abs((B_abs_floor / max(1.0 - slack, 1.0e-300)) ** 2 * da * db - product) / np.maximum(
        product,
        1.0e-300,
    )

    out = df.copy()
    out["jost_required_product"] = required_product
    out["jost_observed_product"] = product
    out["jost_product_margin"] = product_margin
    out["transfer_upper_ratio_with_formula_error"] = transfer_upper_ratio
    out["jost_product_lower_from_transfer"] = product_lower_from_transfer
    out["jost_product_lower_from_transfer_margin"] = product_lower_margin
    out["pole_endpoint_distance_a"] = da
    out["pole_endpoint_distance_b"] = db
    out["pole_endpoint_distance_product"] = da * db
    out["projective_B_abs_from_a"] = B_abs_from_a
    out["projective_B_abs_from_b"] = B_abs_from_b
    out["projective_B_abs_floor"] = B_abs_floor
    out["projective_B_abs_rel_mismatch"] = B_abs_rel_mismatch
    out["pole_product_lower_bound"] = pole_product_lower
    out["pole_product_lower_margin"] = pole_product_margin
    out["pole_product_identity_rel_error"] = pole_identity_rel_error
    out["jost_product_bound_status"] = np.where(
        product_lower_from_transfer > required_product,
        "passes_jost_product_lower_bound",
        "not_lifted_jost_product_bound",
    )
    out["pole_exclusion_status"] = np.where(
        pole_product_lower > required_product,
        "passes_projective_pole_exclusion",
        "not_lifted_projective_pole_exclusion",
    )
    return out


def summary_row(label: str, sub: pd.DataFrame) -> dict[str, float | str]:
    return {
        "group": label,
        "n_rows": float(len(sub)),
        "n_jost_product_pass": float((sub["jost_product_bound_status"] == "passes_jost_product_lower_bound").sum()),
        "n_pole_exclusion_pass": float((sub["pole_exclusion_status"] == "passes_projective_pole_exclusion").sum()),
        "required_product_min": float(sub["jost_required_product"].min()),
        "required_product_median": float(sub["jost_required_product"].median()),
        "required_product_max": float(sub["jost_required_product"].max()),
        "observed_product_min": float(sub["jost_observed_product"].min()),
        "observed_product_median": float(sub["jost_observed_product"].median()),
        "observed_product_max": float(sub["jost_observed_product"].max()),
        "product_margin_min": float(sub["jost_product_margin"].min()),
        "product_margin_median": float(sub["jost_product_margin"].median()),
        "product_margin_max": float(sub["jost_product_margin"].max()),
        "lower_from_transfer_margin_min": float(sub["jost_product_lower_from_transfer_margin"].min()),
        "lower_from_transfer_margin_median": float(sub["jost_product_lower_from_transfer_margin"].median()),
        "pole_product_lower_margin_min": float(sub["pole_product_lower_margin"].min()),
        "pole_product_lower_margin_median": float(sub["pole_product_lower_margin"].median()),
        "pole_endpoint_distance_a_min": float(sub["pole_endpoint_distance_a"].min()),
        "pole_endpoint_distance_b_min": float(sub["pole_endpoint_distance_b"].min()),
        "projective_B_abs_floor_min": float(sub["projective_B_abs_floor"].min()),
        "projective_B_abs_rel_mismatch_max": float(sub["projective_B_abs_rel_mismatch"].max()),
        "pole_product_identity_rel_error_max": float(sub["pole_product_identity_rel_error"].max()),
        "transfer_upper_ratio_max": float(sub["transfer_upper_ratio_with_formula_error"].max()),
        "jost_product_theorem_status": "has_uniform_jost_product_lower_bound"
        if float((sub["jost_product_bound_status"] == "passes_jost_product_lower_bound").sum()) == len(sub)
        else "not_uniform_on_current_grid",
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for D, sub in detail.groupby("D", sort=True):
        rows.append(summary_row(f"D={D:g}", sub))
    rows.append(summary_row("global", detail))
    return pd.DataFrame(rows)


def plot_budget(detail: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))

    ax = axes[0]
    ordered = detail.sort_values("jost_product_lower_from_transfer_margin").reset_index(drop=True)
    x = np.arange(len(ordered))
    ax.scatter(x, ordered["jost_product_margin"], s=28, alpha=0.72, label="observed product")
    ax.scatter(
        x,
        ordered["jost_product_lower_from_transfer_margin"],
        s=24,
        alpha=0.82,
        label="lower bound from transfer envelope",
    )
    ax.scatter(
        x,
        ordered["pole_product_lower_margin"],
        s=18,
        alpha=0.45,
        marker="x",
        label="pole-factor lower bound",
    )
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("rows sorted by lower-bound margin")
    ax.set_ylabel("Jost product / required product")
    ax.set_title("Endpoint Jost-product margin")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    colors = {6.0: "#d95f02", 12.0: "#1b9e77", 18.0: "#386cb0"}
    for D, sub in detail.groupby("D", sort=True):
        ax.scatter(
            sub["pole_endpoint_distance_a"],
            sub["projective_B_abs_floor"],
            s=36,
            alpha=0.78,
            label=f"D={D:g}",
            color=colors.get(float(D), None),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("near-endpoint distance |a-p|")
    ax.set_ylabel("projective |B| floor")
    ax.set_title("Pole exclusion is endpoint-specific")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projective-detail", type=Path, default=DEFAULT_PROJECTIVE)
    ap.add_argument("--pair", default="1600_to_3200")
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--relative-slack", type=float, default=1.0e-6)
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    projective = pd.read_csv(args.projective_detail)
    detail = build_detail(projective=projective, pair=str(args.pair), slack=float(args.relative_slack))
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
