#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/Users/boypatrick/codex/PSLT_quantam")
OUTDIR = ROOT / "output" / "hll_absolute_normalization"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def section_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["section"]: row for row in rows}


def fit_b4(rows: list[dict[str, float]], q4_ref: float) -> float:
    xs = []
    ys = []
    for row in rows:
        x = row["sigma_curv0"] * (row["q4_iso"] - q4_ref)
        y = row["sigma_exact"] - row["sigma_curv0"]
        xs.append(x)
        ys.append(y)
    num = sum(x * y for x, y in zip(xs, ys))
    den = sum(x * x for x in xs)
    return num / den


def eval_fit(rows_all: list[dict[str, float]], b4: float, q4_ref: float) -> tuple[float, float]:
    relerrs = []
    for row in rows_all:
        sigma_fit = row["sigma_curv0"] * (1.0 + b4 * (row["q4_iso"] - q4_ref))
        relerrs.append(abs(sigma_fit / row["sigma_exact"] - 1.0))
    return max(relerrs), sum(relerrs) / len(relerrs)


def main() -> None:
    detail_rows_raw = read_csv(OUTDIR / "hll_sigma_mu_curvature_correction_detail.csv")
    summary_rows = section_map(read_csv(OUTDIR / "hll_sigma_mu_curvature_correction_summary.csv"))

    detail_rows = []
    for row in detail_rows_raw:
        detail_rows.append(
            {
                "D": float(row["D"]),
                "sigma_exact": float(row["sigma_exact"]),
                "q4_iso": float(row["q4_iso"]),
                "sigma_curv0": float(row["sigma_curv0"]),
                "sigma_curv1": float(row["sigma_curv1"]),
                "sigma_curv0_relerr": float(row["sigma_curv0_relerr"]),
                "sigma_curv1_relerr": float(row["sigma_curv1_relerr"]),
                "kernel_rel_sup_curv0": float(row["kernel_rel_sup_curv0"]),
                "kernel_rel_sup_curv1": float(row["kernel_rel_sup_curv1"]),
            }
        )

    q4_ref = float(summary_rows["quartic_invariant_profile"]["q4_ref"])
    b4_full = float(summary_rows["first_quartic_correction"]["b4_fit"])

    def subset(pred):
        return [row for row in detail_rows if pred(row)]

    subsets = {
        "full_window": detail_rows,
        "excluding_D5": subset(lambda row: row["D"] != 5.0),
        "outer_window": subset(lambda row: row["D"] <= 8.0 or row["D"] >= 12.0),
        "left_half": subset(lambda row: row["D"] <= 10.0),
        "right_half": subset(lambda row: row["D"] >= 10.0),
    }

    detail_out = OUTDIR / "hll_b4_tightening_synthesis_detail.csv"
    summary_out = OUTDIR / "hll_b4_tightening_synthesis_summary.csv"

    detail_rows_out: list[dict[str, float | str]] = []
    for name, subrows in subsets.items():
        b4_fit = fit_b4(subrows, q4_ref)
        max_relerr, mean_relerr = eval_fit(detail_rows, b4_fit, q4_ref)
        detail_rows_out.append(
            {
                "subset": name,
                "b4_fit": b4_fit,
                "rel_drift_vs_full": abs(b4_fit - b4_full) / b4_full,
                "max_sigma_relerr_on_full_grid": max_relerr,
                "mean_sigma_relerr_on_full_grid": mean_relerr,
            }
        )

    q4_vals = [row["q4_iso"] for row in detail_rows]
    summary_row = {
        "b4_fit_full": b4_full,
        "q4_ref": q4_ref,
        "q4_min": min(q4_vals),
        "q4_max": max(q4_vals),
        "q4_rel_span": float(summary_rows["quartic_invariant_profile"]["q4_rel_span"]),
        "uncorrected_width_max_relerr": float(summary_rows["uncorrected_curvature_candidate"]["max_sigma_curv0_relerr"]),
        "uncorrected_width_mean_relerr": float(summary_rows["uncorrected_curvature_candidate"]["mean_sigma_curv0_relerr"]),
        "uncorrected_kernel_max_rel_sup": float(summary_rows["uncorrected_curvature_candidate"]["max_kernel_rel_sup_curv0"]),
        "corrected_width_max_relerr": float(summary_rows["first_quartic_correction"]["max_sigma_curv1_relerr"]),
        "corrected_width_mean_relerr": float(summary_rows["first_quartic_correction"]["mean_sigma_curv1_relerr"]),
        "corrected_kernel_max_rel_sup": float(summary_rows["first_quartic_correction"]["max_kernel_rel_sup_curv1"]),
        "width_improvement_factor": float(summary_rows["uncorrected_curvature_candidate"]["max_sigma_curv0_relerr"])
        / float(summary_rows["first_quartic_correction"]["max_sigma_curv1_relerr"]),
        "kernel_improvement_factor": float(summary_rows["uncorrected_curvature_candidate"]["max_kernel_rel_sup_curv0"])
        / float(summary_rows["first_quartic_correction"]["max_kernel_rel_sup_curv1"]),
        "b4_rel_drift_excluding_D5": next(
            row["rel_drift_vs_full"] for row in detail_rows_out if row["subset"] == "excluding_D5"
        ),
        "b4_rel_drift_outer_window": next(
            row["rel_drift_vs_full"] for row in detail_rows_out if row["subset"] == "outer_window"
        ),
        "b4_rel_drift_left_half": next(
            row["rel_drift_vs_full"] for row in detail_rows_out if row["subset"] == "left_half"
        ),
        "b4_rel_drift_right_half": next(
            row["rel_drift_vs_full"] for row in detail_rows_out if row["subset"] == "right_half"
        ),
        "remaining_primary_target": "quartic_response_coefficient_b4_only",
        "secondary_tightening_1": "sub_1e-3_projected_box_polish",
        "secondary_tightening_2": "none_on_normalization_mainline",
    }

    with detail_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows_out)

    with summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Wrote {detail_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
