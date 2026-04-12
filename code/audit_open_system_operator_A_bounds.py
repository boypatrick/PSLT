#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Translate conditional complement log-det bounds sigma_a=-log det(I-A_a) into "
            "true operator consequences for tr(A_a), ||A_a||, the Loewner floor on H_eff, "
            "and the Corollary-1 log-det remainder."
        )
    )
    parser.add_argument(
        "--sigma-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_sigma_offdiag_tolerance_summary.csv"
        ),
    )
    parser.add_argument(
        "--detail-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_A_bounds_detail.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_A_bounds_summary.csv"
        ),
    )
    args = parser.parse_args()

    sigma_df = pd.read_csv(args.sigma_input).copy()
    sigma = np.asarray(sigma_df["sigma_upper_rank1"], dtype=float)

    sigma_df["trA_upper_from_sigma"] = sigma
    sigma_df["opA_upper_from_sigma"] = 1.0 - np.exp(-sigma)
    sigma_df["opA_consistency_residual"] = np.abs(
        np.asarray(sigma_df["opA_upper_from_sigma"], dtype=float)
        - np.asarray(sigma_df["a_upper_rank1"], dtype=float)
    )
    sigma_df["heff_loewner_factor"] = np.exp(-sigma)
    sigma_df["cor1_logdet_gap_upper"] = (
        2.0 * np.asarray(sigma_df["opA_upper_from_sigma"], dtype=float)
        / np.maximum(1.0 - np.asarray(sigma_df["opA_upper_from_sigma"], dtype=float), 1e-30)
    )
    sigma_df["conditional_statement"] = (
        "If the dominant-channel sigma bound is valid, then tr(A_a)<=sigma_a and "
        "||A_a||<=1-exp(-sigma_a), hence H_eff>=exp(-sigma_a) H_PP."
    )

    keep_cols = [
        "objective_tol",
        "lambda_upper",
        "family_g_floor",
        "proxy_name",
        "envelope",
        "xi_upper",
        "a_upper_rank1",
        "sigma_upper_rank1",
        "trA_upper_from_sigma",
        "opA_upper_from_sigma",
        "opA_consistency_residual",
        "heff_loewner_factor",
        "cor1_logdet_gap_upper",
        "lifted_g_floor",
        "separator_min",
        "eps_delta_factor_max",
        "tail_mass_upper",
        "eps0_upper",
        "eps_delta_upper",
        "delta_q_ref",
        "conditional_statement",
    ]
    detail = sigma_df[keep_cols].sort_values(["objective_tol", "sigma_upper_rank1", "proxy_name", "envelope"])

    summary_masks = [
        (detail["proxy_name"] == "xi_exact") & (detail["envelope"] == "exact"),
        (detail["proxy_name"] == "runnerup_blend_abs_action") & (detail["envelope"] == "p95"),
    ]
    summary = detail[np.logical_or.reduce(summary_masks)].copy()
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
