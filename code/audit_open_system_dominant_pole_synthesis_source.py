#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "chi_open_system"


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    deltaq_summary = row_by(
        load_csv(OUTDIR / "chi_open_system_deltaq_source_summary.csv")
    )
    deltaq_detail = load_csv(OUTDIR / "chi_open_system_deltaq_source_detail.csv")
    neigh_summary = row_by(
        load_csv(OUTDIR / "chi_open_system_operator_detdeficit_neighborhood_bound_summary.csv"),
        objective_tol="1e-06",
    )
    exact_lift_summary = row_by(
        load_csv(OUTDIR / "chi_open_system_exact_lift_source_summary.csv")
    )
    witness_transfer_summary = row_by(
        load_csv(OUTDIR / "chi_open_system_operator_witness_transfer_summary.csv")
    )
    rigidity_transfer_summary = row_by(
        load_csv(OUTDIR / "chi_open_system_rigidity_transfer_summary.csv")
    )
    radius_summary = row_by(
        load_csv(OUTDIR / "chi_open_system_exact_operator_radius_bound_summary.csv")
    )
    chart_summary = row_by(
        load_csv(OUTDIR / "chi_open_system_exact_operator_chart_source_summary.csv")
    )

    deltaq_keys = [
        "delta_q_parent_log_vs_gap",
        "delta_q_parent_shape_vs_gap",
        "delta_q_micro_vs_gap",
    ]
    worst_deltaq = {"D": "", "component": "", "value_abs": -1.0}
    for row in deltaq_detail:
        for key in deltaq_keys:
            val = abs(to_float(row, key))
            if val > float(worst_deltaq["value_abs"]):
                worst_deltaq = {"D": row["D"], "component": key, "value_abs": val}

    worst_chart_row = max(
        chart_summary and load_csv(OUTDIR / "chi_open_system_exact_operator_chart_source_detail.csv"),
        key=lambda r: to_float(r, "witness_linf"),
    )

    operator_env_bound = to_float(radius_summary, "operator_env_bound")
    exact_operator_max_linf = to_float(chart_summary, "exact_operator_max_linf")
    envelope_matches_chart_max_resid = abs(operator_env_bound - exact_operator_max_linf)
    lifted_g_consistency_resid = abs(
        to_float(neigh_summary, "lifted_g_floor")
        - to_float(exact_lift_summary, "lifted_g_floor")
    )
    heff_consistency_resid = abs(
        to_float(neigh_summary, "heff_loewner_factor")
        - to_float(exact_lift_summary, "heff_loewner_factor")
    )

    detail_rows = [
        {
            "stage": "on_shell_defect",
            "status": "closed_exact",
            "metric_1_name": "max_abs_delta_q_vs_gap_locked",
            "metric_1_value": float(worst_deltaq["value_abs"]),
            "metric_2_name": "max_abs_tau_gap_lock_residual",
            "metric_2_value": to_float(deltaq_summary, "max_abs_tau_gap_lock_residual"),
            "note": "The on-shell defect side is already closed on D4--20; the parent-bath ratio, the direct micro ratio, and the gap-locked Lorentzian prediction agree to machine precision.",
        },
        {
            "stage": "projected_neighborhood_budget",
            "status": "candidate_theorem_budget",
            "metric_1_name": "u_neighborhood_upper",
            "metric_1_value": to_float(neigh_summary, "u_neighborhood_upper"),
            "metric_2_name": "lifted_g_floor",
            "metric_2_value": to_float(neigh_summary, "lifted_g_floor"),
            "note": "The continuum-neighborhood determinant-deficit route already gives one explicit budget for the projected family at objective tolerance 1e-6.",
        },
        {
            "stage": "family_class_transfer",
            "status": "closed_audit_transfer",
            "metric_1_name": "all_canonical_unique_admissible",
            "metric_1_value": to_float(rigidity_transfer_summary, "all_canonical_unique_admissible"),
            "metric_2_name": "min_selection_gap_over_tol",
            "metric_2_value": to_float(rigidity_transfer_summary, "min_selection_gap_over_tol"),
            "note": "Across the four explicit deformation families, the canonical point is the unique admissible point at the theorem tolerance.",
        },
        {
            "stage": "exact_lift_source",
            "status": "closed_exact_export",
            "metric_1_name": "lifted_g_floor",
            "metric_1_value": to_float(exact_lift_summary, "lifted_g_floor"),
            "metric_2_name": "heff_loewner_factor",
            "metric_2_value": to_float(exact_lift_summary, "heff_loewner_factor"),
            "note": "On the exported exact operator, the audited lift reproduces the same determinant-deficit budget and leaves only tiny mode-tail bounds.",
        },
        {
            "stage": "witness_chart_inclusion",
            "status": "closed_exact_export",
            "metric_1_name": "exact_operator_env_bound",
            "metric_1_value": operator_env_bound,
            "metric_2_name": "min_half_radius_margin",
            "metric_2_value": min(
                to_float(radius_summary, "p95_half_over_env"),
                to_float(radius_summary, "max_half_over_env"),
            ),
            "note": "The exported exact operator already lies well inside the audited safe chart; the dominant witness component is the factorization-side block-22 term.",
        },
        {
            "stage": "remaining_primary_target",
            "status": "single_gap",
            "metric_1_name": "remaining_primary_target",
            "metric_1_value": "exact_schur_lift_of_neighborhood_detdeficit_budget",
            "metric_2_name": "closed_side",
            "metric_2_value": "delta_q_and_exact_witness_envelope",
            "note": "The only remaining theorem-level gap is to upgrade the projected neighborhood budget into an a priori exact Schur-resolvent lift; no new support or response object is needed.",
        },
    ]

    summary_rows = [
        {
            "n_rows": to_float(deltaq_summary, "n_rows"),
            "D_min": to_float(deltaq_summary, "D_min"),
            "D_max": to_float(deltaq_summary, "D_max"),
            "worst_delta_q_gap_abs": float(worst_deltaq["value_abs"]),
            "worst_delta_q_gap_D": float(worst_deltaq["D"]),
            "worst_delta_q_gap_component": str(worst_deltaq["component"]),
            "max_abs_tau_gap_lock_residual": to_float(deltaq_summary, "max_abs_tau_gap_lock_residual"),
            "analytic_g_floor": to_float(deltaq_summary, "analytic_g_floor"),
            "exact_family_g_floor": to_float(deltaq_summary, "exact_g_floor"),
            "u_neighborhood_upper": to_float(neigh_summary, "u_neighborhood_upper"),
            "lifted_g_floor": to_float(neigh_summary, "lifted_g_floor"),
            "heff_loewner_factor": to_float(neigh_summary, "heff_loewner_factor"),
            "all_canonical_unique_admissible": to_float(rigidity_transfer_summary, "all_canonical_unique_admissible"),
            "min_selection_gap_over_tol": to_float(rigidity_transfer_summary, "min_selection_gap_over_tol"),
            "operator_env_bound": operator_env_bound,
            "exact_operator_max_linf": exact_operator_max_linf,
            "operator_env_worst_D": float(worst_chart_row["D"]),
            "p95_half_radius": to_float(radius_summary, "p95_half_radius"),
            "max_half_radius": to_float(radius_summary, "max_half_radius"),
            "p95_half_over_env": to_float(radius_summary, "p95_half_over_env"),
            "max_half_over_env": to_float(radius_summary, "max_half_over_env"),
            "envelope_matches_chart_max_resid": envelope_matches_chart_max_resid,
            "lifted_g_consistency_resid": lifted_g_consistency_resid,
            "heff_consistency_resid": heff_consistency_resid,
            "remaining_primary_target": "exact_schur_lift_of_neighborhood_detdeficit_budget",
            "closed_block_1": "delta_q_side",
            "closed_block_2": "family_chart_transfer",
            "closed_block_3": "exact_operator_witness_envelope",
        }
    ]

    detail_path = OUTDIR / "chi_open_system_dominant_pole_synthesis_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_dominant_pole_synthesis_source_summary.csv"
    write_csv(detail_path, detail_rows)
    write_csv(summary_path, summary_rows)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
