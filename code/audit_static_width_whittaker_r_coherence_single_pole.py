#!/usr/bin/env python3
"""
R-coherence / single-pole necessary-condition audit for the static-width line.

This is the theorem-facing follow-up to S14.  It does not solve for roots, does
not introduce a new backend, and does not alter the Whittaker family.  It asks
only whether the already-certified S8 local boxes can contain one common parent
pole when grouped across the existing interface radii R, matching radii R_tail,
and parity labels.

For a group of local boxes Q_i centered at E_i with half-widths rho_i, a single
parent pole represented by all boxes must lie in the intersection of those
boxes.  The exact axis-aligned common-cover factor for the current equal-width
boxes is

    c_square = max( range(Re E_i)/(2 rho_Re), range(Im E_i)/(2 rho_Im) ).

The unexpanded boxes intersect iff c_square <= 1.  Values above one quantify
the necessary enlargement before a single-pole explanation is even geometrically
available.

Outputs:
  - output/cap_resonance_1d/static_width_whittaker_r_coherence_single_pole_detail.csv
  - output/cap_resonance_1d/static_width_whittaker_r_coherence_single_pole_summary.csv
  - output/cap_resonance_1d/static_width_whittaker_r_coherence_single_pole_budget.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_static_width_siegert_extraction import OUTDIR


DEFAULT_ARGUMENT = OUTDIR / "static_width_whittaker_argument_principle_detail.csv"
DEFAULT_CONTOUR_JOST = OUTDIR / "static_width_whittaker_contour_jost_bound_detail.csv"
DEFAULT_TAG = "static_width_whittaker_r_coherence_single_pole"


def contour_lookup(contour: pd.DataFrame) -> dict[tuple[float, float, float, str], pd.Series]:
    out: dict[tuple[float, float, float, str], pd.Series] = {}
    for _, row in contour.iterrows():
        out[(float(row["D"]), float(row["R"]), float(row["R_tail_root"]), str(row["parity"]))] = row
    return out


def clean_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


def clean_key_label(key: object) -> str:
    if isinstance(key, tuple):
        return "(" + ",".join(clean_value(part) for part in key) + ")"
    return clean_value(key)


def box_group_row(
    family: str,
    label: str,
    sub: pd.DataFrame,
    contour_rows: list[pd.Series],
) -> dict[str, float | str]:
    re = sub["E0_real"].to_numpy(dtype=float)
    im = sub["E0_imag"].to_numpy(dtype=float)
    box_re = sub["box_re"].to_numpy(dtype=float)
    box_im = sub["box_im"].to_numpy(dtype=float)
    E = re + 1j * im

    re_left = re - box_re
    re_right = re + box_re
    im_low = im - box_im
    im_high = im + box_im
    re_overlap = float(np.min(re_right) - np.max(re_left))
    im_overlap = float(np.min(im_high) - np.max(im_low))
    has_intersection = bool(re_overlap >= 0.0 and im_overlap >= 0.0)

    re_span = float(np.max(re) - np.min(re)) if len(re) else 0.0
    im_span = float(np.max(im) - np.min(im)) if len(im) else 0.0
    abs_diam = 0.0
    for z0 in E:
        for z1 in E:
            abs_diam = max(abs_diam, float(abs(z0 - z1)))

    rho_re = float(np.min(box_re))
    rho_im = float(np.min(box_im))
    rho_disk = float(np.min(np.minimum(box_re, box_im)))
    square_cover = max(
        re_span / max(2.0 * rho_re, 1.0e-300),
        im_span / max(2.0 * rho_im, 1.0e-300),
    )
    disk_diameter_lower = abs_diam / max(2.0 * rho_disk, 1.0e-300)

    if contour_rows:
        contour = pd.DataFrame(contour_rows)
        n_contour_pass = float((contour["contour_jost_status"] == "passes_contour_jost_box").sum())
        boundary_ratio_max = float(contour["boundary_ratio_direct_max"].max())
        product_margin_min = float(contour["boundary_product_margin_min"].min())
        winding_abs_max = float(
            max(
                contour["jost_a_winding_round"].abs().max(),
                contour["jost_b_winding_round"].abs().max(),
            )
        )
    else:
        n_contour_pass = 0.0
        boundary_ratio_max = float("nan")
        product_margin_min = float("nan")
        winding_abs_max = float("nan")

    all_contour_pass = bool(n_contour_pass == len(sub))
    if has_intersection and all_contour_pass:
        status = "single_pole_box_intersection_available"
    elif all_contour_pass:
        status = "tail_box_control_but_no_single_pole_intersection"
    else:
        status = "missing_tail_box_control_or_intersection"

    return {
        "D": float(sub["D"].iloc[0]),
        "group_family": family,
        "group_label": label,
        "n_boxes": float(len(sub)),
        "R_values": ",".join(f"{x:g}" for x in sorted(sub["R"].astype(float).unique())),
        "R_tail_values": ",".join(f"{x:g}" for x in sorted(sub["R_tail"].astype(float).unique())),
        "parities": ",".join(str(x) for x in sorted(sub["parity"].astype(str).unique())),
        "re_span": re_span,
        "im_span": im_span,
        "abs_diameter": abs_diam,
        "box_re_min": rho_re,
        "box_im_min": rho_im,
        "re_overlap_width": re_overlap,
        "im_overlap_width": im_overlap,
        "has_common_box_intersection": float(has_intersection),
        "exact_square_cover_factor": float(square_cover),
        "disk_cover_diameter_lower_factor": float(disk_diameter_lower),
        "n_contour_jost_pass": n_contour_pass,
        "boundary_ratio_direct_max": boundary_ratio_max,
        "boundary_product_margin_min": product_margin_min,
        "winding_abs_max": winding_abs_max,
        "single_pole_status": status,
    }


def add_group_rows(
    rows: list[dict[str, float | str]],
    family: str,
    grouped: list[tuple[object, pd.DataFrame]],
    contour_by_key: dict[tuple[float, float, float, str], pd.Series],
) -> None:
    for key, sub in grouped:
        sub = sub.sort_values(["R_tail", "R", "parity"]).reset_index(drop=True)
        contour_rows: list[pd.Series] = []
        for _, row in sub.iterrows():
            lookup_key = (float(row["D"]), float(row["R"]), float(row["R_tail"]), str(row["parity"]))
            if lookup_key in contour_by_key:
                contour_rows.append(contour_by_key[lookup_key])
        label = clean_key_label(key)
        rows.append(box_group_row(family=family, label=label, sub=sub, contour_rows=contour_rows))


def build_detail(argument: pd.DataFrame, contour: pd.DataFrame) -> pd.DataFrame:
    contour_by_key = contour_lookup(contour)
    rows: list[dict[str, float | str]] = []

    add_group_rows(
        rows,
        "all_cases_by_D",
        [(D, sub) for D, sub in argument.groupby("D", sort=True)],
        contour_by_key,
    )
    add_group_rows(
        rows,
        "R_sweep_fixed_tail_parity",
        [((D, R_tail, parity), sub) for (D, R_tail, parity), sub in argument.groupby(["D", "R_tail", "parity"], sort=True)],
        contour_by_key,
    )
    add_group_rows(
        rows,
        "R_sweep_fixed_tail",
        [((D, R_tail), sub) for (D, R_tail), sub in argument.groupby(["D", "R_tail"], sort=True)],
        contour_by_key,
    )
    add_group_rows(
        rows,
        "R_sweep_fixed_parity",
        [((D, parity), sub) for (D, parity), sub in argument.groupby(["D", "parity"], sort=True)],
        contour_by_key,
    )
    add_group_rows(
        rows,
        "fixed_R_tail_parity_block",
        [((D, R), sub) for (D, R), sub in argument.groupby(["D", "R"], sort=True)],
        contour_by_key,
    )
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (family, D), sub in detail.groupby(["group_family", "D"], sort=True):
        rows.append(summary_row(family=family, D=D, sub=sub))
    for family, sub in detail.groupby("group_family", sort=True):
        rows.append(summary_row(family=family, D="global", sub=sub))
    rows.append(summary_row(family="all_group_families", D="global", sub=detail))
    return pd.DataFrame(rows)


def summary_row(family: str, D: object, sub: pd.DataFrame) -> dict[str, float | str]:
    intersections = sub["has_common_box_intersection"] > 0.5
    all_tail_box_control = (
        (sub["n_contour_jost_pass"] == sub["n_boxes"])
        & (sub["boundary_ratio_direct_max"] < 1.0)
        & (sub["winding_abs_max"] == 0.0)
    )
    best = sub.sort_values("exact_square_cover_factor").iloc[0]
    status = (
        "has_single_pole_box_intersection"
        if bool(intersections.any())
        else "no_single_pole_intersection_on_current_boxes"
    )
    if bool(all_tail_box_control.all()) and not bool(intersections.any()):
        status = "tail_box_control_closed_but_single_pole_intersection_fails"
    return {
        "group_family": family,
        "D": D,
        "n_groups": float(len(sub)),
        "n_groups_with_common_intersection": float(intersections.sum()),
        "n_groups_with_tail_box_control": float(all_tail_box_control.sum()),
        "min_exact_square_cover_factor": float(sub["exact_square_cover_factor"].min()),
        "median_exact_square_cover_factor": float(sub["exact_square_cover_factor"].median()),
        "max_exact_square_cover_factor": float(sub["exact_square_cover_factor"].max()),
        "min_disk_cover_diameter_lower_factor": float(sub["disk_cover_diameter_lower_factor"].min()),
        "max_boundary_ratio_direct": float(sub["boundary_ratio_direct_max"].max()),
        "min_boundary_product_margin": float(sub["boundary_product_margin_min"].min()),
        "best_group_label": str(best["group_label"]),
        "best_group_R_values": str(best["R_values"]),
        "best_group_R_tail_values": str(best["R_tail_values"]),
        "best_group_parities": str(best["parities"]),
        "r_coherence_status": status,
    }


def plot_budget(detail: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [
        "all_cases_by_D",
        "R_sweep_fixed_tail_parity",
        "R_sweep_fixed_tail",
        "R_sweep_fixed_parity",
        "fixed_R_tail_parity_block",
    ]
    labels = [
        "all",
        "R sweep\nfixed tail/parity",
        "R sweep\nfixed tail",
        "R sweep\nfixed parity",
        "fixed R\ntail/parity",
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    positions = np.arange(len(families))
    data = [detail[detail["group_family"] == family]["exact_square_cover_factor"].to_numpy(dtype=float) for family in families]
    ax.boxplot(data, positions=positions, widths=0.58, showfliers=True)
    for pos, vals in zip(positions, data):
        ax.scatter(np.full_like(vals, pos, dtype=float), vals, s=22, alpha=0.65)
    ax.axhline(1.0, color="black", lw=1.0, ls="--", label="common-box threshold")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("exact square cover factor")
    ax.set_title("Single-pole common-box necessary condition")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--argument-detail", type=Path, default=DEFAULT_ARGUMENT)
    ap.add_argument("--contour-jost-detail", type=Path, default=DEFAULT_CONTOUR_JOST)
    ap.add_argument("--out-tag", default=DEFAULT_TAG)
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    argument = pd.read_csv(args.argument_detail)
    contour = pd.read_csv(args.contour_jost_detail)
    detail = build_detail(argument=argument, contour=contour)
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
