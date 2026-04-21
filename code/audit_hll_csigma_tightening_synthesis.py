#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path("/Users/boypatrick/codex/PSLT_quantam")
OUTDIR = ROOT / "output" / "hll_absolute_normalization"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_summary_by_section(path: Path) -> dict[str, dict[str, str]]:
    rows = load_rows(path)
    return {row["section"]: row for row in rows}


def fmean(values: list[float]) -> float:
    return sum(values) / len(values)


def rel_span(values: list[float]) -> float:
    mean = fmean(values)
    return (max(values) - min(values)) / mean if mean else 0.0


def main() -> None:
    eymh_detail = load_rows(OUTDIR / "hll_csigma_eymh_core_source_detail.csv")
    sigma_summary = load_summary_by_section(
        OUTDIR / "hll_sigma_mu_curvature_correction_summary.csv"
    )
    csigma_summary = load_summary_by_section(OUTDIR / "hll_csigma_source_summary.csv")
    projected_summary = load_summary_by_section(
        OUTDIR / "hll_csigma_projected_box_source_summary.csv"
    )
    projected_overlap_summary = load_summary_by_section(
        OUTDIR / "hll_csigma_projected_overlap_summary.csv"
    )

    detail_out = OUTDIR / "hll_csigma_tightening_synthesis_detail.csv"
    summary_out = OUTDIR / "hll_csigma_tightening_synthesis_summary.csv"

    mirror_factorization_residuals: list[float] = []
    linearized_relerrs: list[float] = []
    mirror_multipliers: list[float] = []
    exact_values: list[float] = []
    analytic_values: list[float] = []
    self_values: list[float] = []
    mirror_fracs: list[float] = []
    detail_rows: list[dict[str, float]] = []

    for row in eymh_detail:
        D = float(row["D"])
        c_exact = float(row["c_sigma_exact"])
        c_analytic = float(row["c_sigma_analytic"])
        c_self = float(row["c_sigma_self"])
        mirror_fraction = float(row["mirror_fraction"])

        mirror_multiplier = (1.0 + mirror_fraction) ** 0.25
        c_analytic_factored = c_self * mirror_multiplier
        c_analytic_linear = c_self * (1.0 + 0.25 * mirror_fraction)

        factorization_residual = abs(c_analytic - c_analytic_factored)
        linearized_relerr = abs(c_analytic_linear - c_analytic) / c_analytic

        mirror_factorization_residuals.append(factorization_residual)
        linearized_relerrs.append(linearized_relerr)
        mirror_multipliers.append(mirror_multiplier)
        exact_values.append(c_exact)
        analytic_values.append(c_analytic)
        self_values.append(c_self)
        mirror_fracs.append(mirror_fraction)

        detail_rows.append(
            {
                "D": D,
                "c_sigma_exact": c_exact,
                "c_sigma_self": c_self,
                "mirror_fraction": mirror_fraction,
                "mirror_multiplier": mirror_multiplier,
                "c_sigma_analytic": c_analytic,
                "c_sigma_analytic_factored": c_analytic_factored,
                "factorization_residual": factorization_residual,
                "c_sigma_analytic_linearized": c_analytic_linear,
                "linearized_relerr": linearized_relerr,
                "relerr_csigma_analytic": float(row["relerr_csigma_analytic"]),
                "relerr_csigma_self": float(row["relerr_csigma_self"]),
            }
        )

    projected_candidates = projected_summary["candidate_constants"]
    projected_self = projected_summary["self_core_candidate_metrics"]
    projected_analytic = projected_summary["analytic_two_center_candidate_metrics"]
    projected_overlap_candidates = projected_overlap_summary["projected_overlap_selector"]

    sigma_bridge = sigma_summary["bridge_constant_invariance"]
    sigma_quartic = sigma_summary["first_quartic_correction"]
    sigma_uncorrected = sigma_summary["uncorrected_curvature_candidate"]

    selector_bridge = csigma_summary["bridge_constant_candidates"]
    selector_ref = csigma_summary["reference_calibration_metrics"]
    selector_amp = csigma_summary["best_amplitude_fit_metrics"]

    analytic_ref = float(
        load_summary_by_section(OUTDIR / "hll_csigma_eymh_core_source_summary.csv")[
            "reference_point"
        ]["c_sigma_analytic_ref"]
    )

    summary_row = {
        "mirror_factorization_residual_max": max(mirror_factorization_residuals),
        "mirror_multiplier_min": min(mirror_multipliers),
        "mirror_multiplier_max": max(mirror_multipliers),
        "mirror_multiplier_rel_span": rel_span(mirror_multipliers),
        "mirror_fraction_min": min(mirror_fracs),
        "mirror_fraction_max": max(mirror_fracs),
        "linearized_max_relerr": max(linearized_relerrs),
        "c_sigma_exact_min": min(exact_values),
        "c_sigma_exact_max": max(exact_values),
        "c_sigma_exact_rel_span": rel_span(exact_values),
        "c_sigma_analytic_max_relerr_vs_exact": max(
            row["relerr_csigma_analytic"] for row in detail_rows
        ),
        "c_sigma_self_max_relerr_vs_exact": max(
            row["relerr_csigma_self"] for row in detail_rows
        ),
        "c_sigma_ref": float(selector_bridge["c_sigma_ref"]),
        "c_sigma_amp": float(selector_bridge["c_sigma_best_amp"]),
        "c_sigma_projected": float(projected_candidates["c_sigma_projected"]),
        "c_sigma_self": self_values[0],
        "c_sigma_analytic_ref": analytic_ref,
        "rel_drift_analytic_ref_vs_ref": abs(analytic_ref - float(selector_bridge["c_sigma_ref"]))
        / float(selector_bridge["c_sigma_ref"]),
        "rel_drift_projected_vs_amp": float(
            projected_overlap_candidates["rel_drift_proj_global_vs_amp"]
        ),
        "rel_drift_amp_vs_ref": float(selector_bridge["rel_drift_best_amp_vs_ref"]),
        "projected_box_self_log_rmse": float(projected_self["y2_log_rmse"]),
        "projected_box_self_max_rel_y2_resid": float(
            projected_self["max_rel_y2_resid"]
        ),
        "projected_box_analytic_log_rmse": float(projected_analytic["y2_log_rmse"]),
        "projected_box_analytic_max_rel_y2_resid": float(
            projected_analytic["max_rel_y2_resid"]
        ),
        "projected_box_max_rel_improvement_factor": float(
            projected_self["max_rel_y2_resid"]
        )
        / float(projected_analytic["max_rel_y2_resid"]),
        "projected_box_log_rmse_improvement_factor": float(projected_self["y2_log_rmse"])
        / float(projected_analytic["y2_log_rmse"]),
        "uncorrected_width_max_relerr": float(sigma_uncorrected["max_sigma_curv0_relerr"]),
        "quartic_width_max_relerr": float(sigma_quartic["max_sigma_curv1_relerr"]),
        "quartic_kernel_max_rel_sup": float(sigma_quartic["max_kernel_rel_sup_curv1"]),
        "quartic_width_improvement_factor": float(sigma_uncorrected["max_sigma_curv0_relerr"])
        / float(sigma_quartic["max_sigma_curv1_relerr"]),
        "selector_reference_width_log_rmse": float(selector_ref["width_log_rmse"]),
        "selector_amplitude_max_kernel_rel_sup": float(selector_amp["max_kernel_rel_sup"]),
        "remaining_primary_target": "one_center_EYMH_core_constant_with_analytic_mirror_multiplier",
        "secondary_tightening_1": "quartic_response_coefficient_b4",
        "secondary_tightening_2": "sub-1e-3_projected_box_benchmark_polish",
    }

    with detail_out.open("w", newline="") as f:
        fieldnames = list(detail_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    with summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Wrote {detail_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
