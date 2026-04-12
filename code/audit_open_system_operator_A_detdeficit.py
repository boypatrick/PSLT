#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Translate a 2x2 determinant-deficit budget u_a = 1-det(I-A_a) into "
            "true operator bounds for a positive contraction A_a: ||A_a||, tr(A_a), "
            "the Loewner floor on H_eff, and the Corollary-1 log-det remainder."
        )
    )
    parser.add_argument(
        "--source-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_sigma_offdiag_tolerance_summary.csv"
        ),
    )
    parser.add_argument(
        "--detail-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_A_detdeficit_detail.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_A_detdeficit_summary.csv"
        ),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.source_input).copy()
    u = np.asarray(df["a_upper_rank1"], dtype=float)

    # For any 2x2 positive contraction A with eigenvalues alpha_1, alpha_2 in [0,1),
    # u := 1 - det(I-A) = alpha_1 + alpha_2 - alpha_1 alpha_2.
    # Then ||A|| <= u and tr(A) <= 2 * (1 - sqrt(1-u)).
    df["det_deficit_upper"] = u
    df["opA_upper_from_detdeficit"] = u
    df["trA_upper_from_detdeficit"] = 2.0 * (1.0 - np.sqrt(np.maximum(1.0 - u, 0.0)))
    df["sigma_upper_from_detdeficit"] = -np.log(np.maximum(1.0 - u, 1e-30))
    df["heff_loewner_factor_from_detdeficit"] = np.maximum(1.0 - u, 0.0)
    df["cor1_logdet_gap_upper_from_detdeficit"] = (
        2.0 * u / np.maximum(1.0 - u, 1e-30)
    )
    df["trA_vs_sigma_gain"] = np.asarray(df["sigma_upper_from_detdeficit"], dtype=float) - np.asarray(
        df["trA_upper_from_detdeficit"], dtype=float
    )
    df["conditional_statement"] = (
        "If 1-det(I-A_a)<=u_a holds for a 2x2 positive contraction, then ||A_a||<=u_a and "
        "tr(A_a)<=2*(1-sqrt(1-u_a)), hence H_eff>=(1-u_a) H_PP."
    )

    keep_cols = [
        "objective_tol",
        "lambda_upper",
        "family_g_floor",
        "proxy_name",
        "envelope",
        "xi_upper",
        "a_upper_rank1",
        "det_deficit_upper",
        "opA_upper_from_detdeficit",
        "trA_upper_from_detdeficit",
        "sigma_upper_from_detdeficit",
        "trA_vs_sigma_gain",
        "heff_loewner_factor_from_detdeficit",
        "cor1_logdet_gap_upper_from_detdeficit",
        "lifted_g_floor",
        "separator_min",
        "eps_delta_factor_max",
        "tail_mass_upper",
        "eps0_upper",
        "eps_delta_upper",
        "delta_q_ref",
        "conditional_statement",
    ]
    detail = df[keep_cols].sort_values(["objective_tol", "det_deficit_upper", "proxy_name", "envelope"])

    mask = (
        ((detail["proxy_name"] == "xi_exact") & (detail["envelope"] == "exact"))
        | ((detail["proxy_name"] == "runnerup_blend_abs_action") & (detail["envelope"] == "p95"))
    )
    summary = detail[mask].copy()
    summary["summary_role"] = np.where(
        summary["proxy_name"] == "xi_exact",
        "exact canonical xi",
        "conservative raw offdiag envelope",
    )
    summary = summary.sort_values(["objective_tol", "summary_role"])

    for out in [args.detail_output, args.summary_output]:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print(args.detail_output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
