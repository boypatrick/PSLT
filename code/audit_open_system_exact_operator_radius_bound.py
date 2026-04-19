#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a conservative exact-operator witness-radius bound directly from the "
            "factorization/statement/cocycle/potential witness equations."
        )
    )
    ap.add_argument(
        "--factorization-summary",
        default=str(OUTDIR / "chi_open_system_bath_factorization_summary.csv"),
    )
    ap.add_argument(
        "--statement-summary",
        default=str(OUTDIR / "chi_open_system_parent_bath_statement_summary.csv"),
    )
    ap.add_argument(
        "--cocycle-summary",
        default=str(OUTDIR / "chi_open_system_parent_bath_cocycle_summary.csv"),
    )
    ap.add_argument(
        "--potential-summary",
        default=str(OUTDIR / "chi_open_system_parent_bath_potential_summary.csv"),
    )
    ap.add_argument(
        "--witness-transfer-summary",
        default=str(OUTDIR / "chi_open_system_operator_witness_transfer_summary.csv"),
    )
    ap.add_argument(
        "--chart-source-summary",
        default=str(OUTDIR / "chi_open_system_exact_operator_chart_source_summary.csv"),
    )
    ap.add_argument(
        "--detail-output",
        default=str(OUTDIR / "chi_open_system_exact_operator_radius_bound_detail.csv"),
    )
    ap.add_argument(
        "--summary-output",
        default=str(OUTDIR / "chi_open_system_exact_operator_radius_bound_summary.csv"),
    )
    args = ap.parse_args()

    fac = pd.read_csv(args.factorization_summary).iloc[0]
    stmt = pd.read_csv(args.statement_summary).iloc[0]
    coc = pd.read_csv(args.cocycle_summary).iloc[0]
    pot = pd.read_csv(args.potential_summary).iloc[0]
    wit = pd.read_csv(args.witness_transfer_summary).iloc[0]
    chart = pd.read_csv(args.chart_source_summary).iloc[0]

    cocycle_mix_env = max(
        float(coc["max_abs_local_generator_residual_mix"]),
        float(coc["max_abs_pairwise_cocycle_residual_mix"]),
        float(coc["max_abs_triangle_defect_mix"]),
    )
    potential_mix_env = max(
        float(pot["max_abs_potential_residual_mix"]),
        float(pot["max_abs_anchor_recovery_residual_mix"]),
        float(pot["max_abs_chain_recovery_residual_mix"]),
    )

    detail = pd.DataFrame(
        [
            {
                "component": "block11",
                "equation_source": "factorization_phi_identifiable",
                "envelope": float(fac["max_abs_bath_norm_phi_residual_identifiable"]),
            },
            {
                "component": "block22",
                "equation_source": "factorization_mix",
                "envelope": float(fac["max_abs_bath_norm_mix_residual"]),
            },
            {
                "component": "trace",
                "equation_source": "statement_trace",
                "envelope": float(stmt["max_abs_parent_trace_residual"]),
            },
            {
                "component": "det",
                "equation_source": "statement_det",
                "envelope": float(stmt["max_abs_parent_det_residual"]),
            },
            {
                "component": "anisotropy",
                "equation_source": "statement_log_anisotropy_identifiable",
                "envelope": float(
                    stmt["max_abs_parent_log_rate_anisotropy_residual_identifiable"]
                ),
            },
            {
                "component": "aux_mix_cocycle",
                "equation_source": "cocycle_mix_envelope",
                "envelope": cocycle_mix_env,
            },
            {
                "component": "aux_mix_potential",
                "equation_source": "potential_mix_envelope",
                "envelope": potential_mix_env,
            },
        ]
    ).sort_values("envelope", ascending=False, ignore_index=True)

    operator_env_bound = float(detail["envelope"].max())
    p95_half_radius = 0.5 * float(wit["p95_safe_radius"])
    max_half_radius = 0.5 * float(wit["max_safe_radius"])

    summary = pd.DataFrame(
        [
            {
                "operator_env_bound": operator_env_bound,
                "dominant_component": str(detail.loc[0, "component"]),
                "dominant_equation_source": str(detail.loc[0, "equation_source"]),
                "exact_operator_max_linf": float(chart["exact_operator_max_linf"]),
                "exact_operator_p95_linf": float(chart["exact_operator_p95_linf"]),
                "env_over_exact_max_linf": operator_env_bound
                / max(float(chart["exact_operator_max_linf"]), 1e-30),
                "p95_half_radius": p95_half_radius,
                "max_half_radius": max_half_radius,
                "p95_half_over_env": p95_half_radius / max(operator_env_bound, 1e-30),
                "max_half_over_env": max_half_radius / max(operator_env_bound, 1e-30),
                "chart_max_over_env": float(chart["exact_operator_max_linf"])
                / max(operator_env_bound, 1e-30),
                "chart_p95_over_env": float(chart["exact_operator_p95_linf"])
                / max(operator_env_bound, 1e-30),
                "block11_env": float(
                    fac["max_abs_bath_norm_phi_residual_identifiable"]
                ),
                "block22_env": float(fac["max_abs_bath_norm_mix_residual"]),
                "trace_env": float(stmt["max_abs_parent_trace_residual"]),
                "det_env": float(stmt["max_abs_parent_det_residual"]),
                "anisotropy_env": float(
                    stmt["max_abs_parent_log_rate_anisotropy_residual_identifiable"]
                ),
                "cocycle_mix_env": cocycle_mix_env,
                "potential_mix_env": potential_mix_env,
            }
        ]
    )

    for out in [args.detail_output, args.summary_output]:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print(args.detail_output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
