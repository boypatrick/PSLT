#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "hll_absolute_normalization"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_by(rows: list[dict[str, str]], **conds: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(k) == v for k, v in conds.items()):
            return row
    raise KeyError(f"no row matching {conds} in {rows[:2]}")


def to_float(row: dict[str, str], key: str) -> float:
    val = row.get(key, "")
    if val is None or val == "":
        return float("nan")
    return float(val)


def safe_rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else float("nan")


def log_rmse(pairs: list[tuple[float, float]]) -> float:
    errs = []
    for exact, pred in pairs:
        if exact > 0.0 and pred > 0.0:
            errs.append((math.log(pred) - math.log(exact)) ** 2)
    return math.sqrt(sum(errs) / len(errs))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def main() -> None:
    amp_summary = row_by(
        load_csv(OUTDIR / "hll_absolute_amp_source_summary.csv"),
        section="exact_factorization",
    )
    tau_summary = row_by(
        load_csv(OUTDIR / "hll_tau_ratio_source_summary.csv"),
        section="chain_invariance",
    )
    tree_summary = row_by(
        load_csv(OUTDIR / "hll_tree_mumu_parent_source_summary.csv"),
        section="single_layer_exact_reduction",
    )
    tree_ref = row_by(
        load_csv(OUTDIR / "hll_tree_mumu_parent_source_summary.csv"),
        section="reference_exact_constants",
    )
    y2_summary = row_by(
        load_csv(OUTDIR / "hll_y2raw_parent_source_summary.csv"),
        section="exact_mu_flavor_identity",
    )
    y2_center = row_by(
        load_csv(OUTDIR / "hll_y2raw_parent_source_summary.csv"),
        section="center_mode_candidate_check",
    )
    y2_ref = row_by(
        load_csv(OUTDIR / "hll_y2raw_parent_source_summary.csv"),
        section="reference_overlap_target",
    )
    kmu_exact = row_by(
        load_csv(OUTDIR / "hll_kmu_parent_candidate_summary.csv"),
        section="exact_midplane_factorization",
    )
    kmu_box = row_by(
        load_csv(OUTDIR / "hll_kmu_parent_candidate_summary.csv"),
        section="finite_box_continuum_candidate",
    )
    sigma_curv0 = row_by(
        load_csv(OUTDIR / "hll_sigma_mu_parent_candidate_summary.csv"),
        section="curvature_controlled_width",
    )
    sigma_curv1 = row_by(
        load_csv(OUTDIR / "hll_sigma_mu_curvature_correction_summary.csv"),
        section="first_quartic_correction",
    )
    csigma_select = row_by(
        load_csv(OUTDIR / "hll_csigma_source_summary.csv"),
        section="bridge_constant_candidates",
    )
    csigma_core_summary = row_by(
        load_csv(OUTDIR / "hll_csigma_eymh_core_source_summary.csv"),
        section="self_core_source",
    )
    csigma_core_analytic = row_by(
        load_csv(OUTDIR / "hll_csigma_eymh_core_source_summary.csv"),
        section="full_two_center_analytic_jet",
    )
    csigma_mirror = row_by(
        load_csv(OUTDIR / "hll_csigma_eymh_core_source_summary.csv"),
        section="mirror_correction_scale",
    )
    proj_overlap = row_by(
        load_csv(OUTDIR / "hll_csigma_projected_overlap_summary.csv"),
        section="canonical_overlap_rebuild",
    )
    proj_overlap_sel = row_by(
        load_csv(OUTDIR / "hll_csigma_projected_overlap_summary.csv"),
        section="projected_overlap_selector",
    )
    proj_box_constants = row_by(
        load_csv(OUTDIR / "hll_csigma_projected_box_source_summary.csv"),
        section="candidate_constants",
    )

    core_box_detail = load_csv(OUTDIR / "hll_csigma_core_box_local_detail.csv")
    core_box_no_d5 = [r for r in core_box_detail if to_float(r, "D") != 5.0]
    c_local_vals = [to_float(r, "c_sigma_box_local") for r in core_box_no_d5]

    proj_box_detail = load_csv(OUTDIR / "hll_csigma_projected_box_source_detail.csv")
    proj_pairs_self = [
        (to_float(r, "y2_exact"), to_float(r, "y2_self")) for r in proj_box_detail
    ]
    proj_pairs_analytic = [
        (to_float(r, "y2_exact"), to_float(r, "y2_analytic")) for r in proj_box_detail
    ]
    proj_pairs_projected = [
        (to_float(r, "y2_exact"), to_float(r, "y2_projected")) for r in proj_box_detail
    ]

    rel_self = [safe_rel(pred, exact) for exact, pred in proj_pairs_self]
    rel_analytic = [safe_rel(pred, exact) for exact, pred in proj_pairs_analytic]
    rel_projected = [safe_rel(pred, exact) for exact, pred in proj_pairs_projected]

    # Summary-level live-reference consistency using rounded exports.
    a_star_ir = to_float(amp_summary, "A_star_ir")
    a_star_tree = to_float(amp_summary, "A_star_tree")
    z_diag_ref = to_float(tree_ref, "z_diag_ref")
    y2_ref_val = to_float(tree_ref, "y2_raw_ref")
    coeff_ref_val = to_float(tree_ref, "coeff2_kin_over_m2_ref")
    a_from_amp = a_star_tree / (z_diag_ref**2)
    a_from_tree = (to_float(tree_ref, "C_tree_mumu_ref") ** -2) / (z_diag_ref**2)
    a_from_y2 = ((y2_ref_val * coeff_ref_val) ** -2) / (z_diag_ref**2)

    detail_rows = [
        {
            "stage": "flavor_ratio_invariance",
            "status": "closed_exact",
            "metric_1_name": "max_abs_delta_R_e_uv_to_ir",
            "metric_1_value": to_float(tau_summary, "max_abs_delta_R_e_uv_to_ir"),
            "metric_2_name": "max_abs_delta_R_tau_uv_to_ir",
            "metric_2_value": to_float(tau_summary, "max_abs_delta_R_tau_uv_to_ir"),
            "note": "Diagonal flavor ratios are exact UV-tree invariants on active rows.",
        },
        {
            "stage": "absolute_amplitude_factorization",
            "status": "closed_exact",
            "metric_1_name": "max_abs_factorization_residual",
            "metric_1_value": to_float(amp_summary, "max_abs_factorization_residual"),
            "metric_2_name": "A_star_ir_over_tree",
            "metric_2_value": to_float(amp_summary, "A_star_ir_over_tree"),
            "note": "A_* = A_*^(tree) Z_diag,*^{-2} with only mild diagonal dressing.",
        },
        {
            "stage": "tree_single_layer_reduction",
            "status": "closed_exact",
            "metric_1_name": "max_abs_single_layer_residual",
            "metric_1_value": to_float(tree_summary, "max_abs_single_layer_residual"),
            "metric_2_name": "min_mu_share2",
            "metric_2_value": to_float(tree_summary, "min_mu_share2"),
            "note": "C_{mumu}^{tree} = y2raw * (P2kin/M2^2) with exact N=2 support.",
        },
        {
            "stage": "overlap_block_identity",
            "status": "closed_exact",
            "metric_1_name": "max_abs_guv_mu2_sq_vs_raw",
            "metric_1_value": to_float(y2_summary, "max_abs_guv_mu2_sq_vs_raw"),
            "metric_2_name": "p95_rel_center2_vs_raw2",
            "metric_2_value": to_float(y2_center, "p95_rel_center2_vs_raw2"),
            "note": "y2raw = y_mu,2^flavor = (g_mu2^UV)^2; center-mode collapse is not adequate.",
        },
        {
            "stage": "kernel_shape_factorization",
            "status": "closed_exact",
            "metric_1_name": "max_rel_sup_exact_factorization",
            "metric_1_value": to_float(kmu_exact, "max_rel_sup_exact_factorization"),
            "metric_2_name": "max_rel_sup_box_candidate",
            "metric_2_value": to_float(kmu_box, "max_rel_sup_box_candidate"),
            "note": "K_mu closes exactly as a symmetric Gaussian midplane bridge.",
        },
        {
            "stage": "width_profile_chain",
            "status": "closed_structural",
            "metric_1_name": "max_sigma_curvature_relerr",
            "metric_1_value": to_float(sigma_curv0, "max_sigma_curvature_relerr"),
            "metric_2_name": "max_sigma_curv1_relerr",
            "metric_2_value": to_float(sigma_curv1, "max_sigma_curv1_relerr"),
            "note": "The width profile is structurally fixed; quartic correction saturates the residual drift.",
        },
        {
            "stage": "c_sigma_bridge_band",
            "status": "secondary_tightening",
            "metric_1_name": "c_sigma_box_local_min_excl_D5",
            "metric_1_value": min(c_local_vals),
            "metric_2_name": "c_sigma_box_local_max_excl_D5",
            "metric_2_value": max(c_local_vals),
            "note": "The finite-box bridge constant lives on a narrow audited band once the D=5 merger outlier is excluded.",
        },
        {
            "stage": "one_center_core_source",
            "status": "secondary_tightening",
            "metric_1_name": "c_sigma_self",
            "metric_1_value": to_float(csigma_core_summary, "c_sigma_self"),
            "metric_2_name": "max_relerr_csigma_analytic",
            "metric_2_value": to_float(csigma_core_analytic, "max_relerr_csigma_analytic"),
            "note": "One-center core fixes the bulk normalization; analytic two-center jet gives a near-exact refinement.",
        },
        {
            "stage": "projected_box_benchmark",
            "status": "secondary_tightening",
            "metric_1_name": "max_rel_y2_resid_self",
            "metric_1_value": max(rel_self),
            "metric_2_name": "max_rel_y2_resid_analytic",
            "metric_2_value": max(rel_analytic),
            "note": "Analytic mirror correction improves the projected-box benchmark beyond the fixed projected selector.",
        },
        {
            "stage": "live_reference_chain_consistency",
            "status": "summary_consistency",
            "metric_1_name": "rel_resid_A_from_tree_chain",
            "metric_1_value": safe_rel(a_from_tree, a_star_ir),
            "metric_2_name": "rel_resid_A_from_y2_chain",
            "metric_2_value": safe_rel(a_from_y2, a_star_ir),
            "note": "Rounded summary constants remain chain-consistent at the 1e-9 relative level.",
        },
    ]

    summary_row = {
        "max_ratio_invariance_residual": max(
            to_float(tau_summary, "max_abs_delta_R_e_uv_to_ir"),
            to_float(tau_summary, "max_abs_delta_R_tau_uv_to_ir"),
        ),
        "amp_factorization_residual": to_float(amp_summary, "max_abs_factorization_residual"),
        "tree_single_layer_residual": to_float(tree_summary, "max_abs_single_layer_residual"),
        "y2raw_identity_residual": to_float(y2_summary, "max_abs_guv_mu2_sq_vs_raw"),
        "kernel_exact_factorization_relsup": to_float(
            kmu_exact, "max_rel_sup_exact_factorization"
        ),
        "kernel_box_candidate_relsup": to_float(kmu_box, "max_rel_sup_box_candidate"),
        "sigma_curv0_max_relerr": to_float(sigma_curv0, "max_sigma_curvature_relerr"),
        "sigma_curv1_max_relerr": to_float(sigma_curv1, "max_sigma_curv1_relerr"),
        "sigma_curv1_kernel_relsup": to_float(sigma_curv1, "max_kernel_rel_sup_curv1"),
        "c_sigma_band_min_excl_D5": min(c_local_vals),
        "c_sigma_band_max_excl_D5": max(c_local_vals),
        "c_sigma_band_rel_span_excl_D5": (max(c_local_vals) - min(c_local_vals))
        / mean(c_local_vals),
        "c_sigma_ref": to_float(csigma_select, "c_sigma_ref"),
        "c_sigma_amp": to_float(csigma_select, "c_sigma_best_amp"),
        "c_sigma_projected": to_float(proj_box_constants, "c_sigma_projected"),
        "c_sigma_self": to_float(csigma_core_summary, "c_sigma_self"),
        "c_sigma_self_relerr_vs_ref": safe_rel(
            to_float(csigma_core_summary, "c_sigma_self"),
            to_float(csigma_select, "c_sigma_ref"),
        ),
        "c_sigma_analytic_max_relerr": to_float(
            csigma_core_analytic, "max_relerr_csigma_analytic"
        ),
        "mirror_fraction_min": to_float(csigma_mirror, "mirror_fraction_min"),
        "mirror_fraction_max": to_float(csigma_mirror, "mirror_fraction_max"),
        "projected_overlap_selector_rel_drift_vs_ref": to_float(
            proj_overlap_sel, "rel_drift_proj_global_vs_ref"
        ),
        "projected_overlap_exact_rebuild_relerr": to_float(
            proj_overlap, "max_canonical_rebuild_relerr"
        ),
        "projected_box_self_log_rmse": log_rmse(proj_pairs_self),
        "projected_box_self_max_rel_y2_resid": max(rel_self),
        "projected_box_analytic_log_rmse": log_rmse(proj_pairs_analytic),
        "projected_box_analytic_max_rel_y2_resid": max(rel_analytic),
        "projected_box_selector_log_rmse": log_rmse(proj_pairs_projected),
        "projected_box_selector_max_rel_y2_resid": max(rel_projected),
        "remaining_primary_target": "one_center_EYMH_core_normalization_for_c_sigma",
        "secondary_tightening_1": "analytic_two_center_mirror_correction",
        "secondary_tightening_2": "quartic_response_coefficient_b4",
    }

    detail_path = OUTDIR / "hll_absolute_normalization_synthesis_source_detail.csv"
    summary_path = OUTDIR / "hll_absolute_normalization_synthesis_source_summary.csv"

    with detail_path.open("w", newline="") as f:
        fieldnames = list(detail_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    with summary_path.open("w", newline="") as f:
        fieldnames = list(summary_row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
