#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def lorentz_factor(r: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + r * r)


def parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def rowwise_g(alpha: float, beta: float, s_part: np.ndarray, s_schur: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.exp(alpha * s_part) + np.exp(beta * s_schur) - 1.0, 0.0))


def rowwise_grad_norm(alpha: float, beta: float, s_part: np.ndarray, s_schur: np.ndarray) -> np.ndarray:
    g = np.maximum(rowwise_g(alpha, beta, s_part, s_schur), 1e-30)
    dga = np.exp(alpha * s_part) * s_part / (2.0 * g)
    dgb = np.exp(beta * s_schur) * s_schur / (2.0 * g)
    return np.sqrt(dga * dga + dgb * dgb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Turn the parent-side proxy gap into a local operator-family lower bound "
            "using the exact parent mismatch functional and the stationarity Hessian."
        )
    )
    parser.add_argument(
        "--parent-blockdet-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_action_loop_eymh_parent_blockdet_audit_map_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--parent-scan-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_action_loop_eymh_parent_kernel_statement_scan_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--stationarity-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_action_loop_eymh_stationarity_audit_summary_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--open-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_micro_bridge_map.csv"
        ),
    )
    parser.add_argument(
        "--delta-q-ref",
        type=float,
        default=1e-3,
        help="Reference on-shell defect used to translate a gap floor into a remainder estimate.",
    )
    parser.add_argument(
        "--objective-tols",
        default="1.500329e-7,5e-7,1e-6,2e-6,2.5e-6",
        help="Comma-separated parent-objective tolerances for the local admissible family.",
    )
    parser.add_argument("--alpha-radius", type=float, default=0.1)
    parser.add_argument("--beta-radius", type=float, default=0.1)
    parser.add_argument("--lambda-radius", type=float, default=0.2)
    parser.add_argument(
        "--detail-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_gap_lower_bound_detail.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_gap_lower_bound_summary.csv"
        ),
    )
    parser.add_argument(
        "--tolerance-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_gap_lower_bound_tolerance_summary.csv"
        ),
    )
    args = parser.parse_args()

    parent_df = pd.read_csv(args.parent_blockdet_input)
    parent_df = parent_df[parent_df["parent_blockdet_action"] > 1e-12].copy()
    parent_df["s_part"] = np.log(parent_df["parent_blockdet_k11"])
    parent_df["s_schur"] = np.log(parent_df["parent_blockdet_schur_norm"])
    parent_df["g_proxy_canonical"] = np.sqrt(
        np.maximum(parent_df["parent_blockdet_k11"] + parent_df["parent_blockdet_schur_norm"] - 1.0, 0.0)
    )

    scan = pd.read_csv(args.parent_scan_input)
    stat = pd.read_csv(args.stationarity_input).iloc[0]
    open_df = pd.read_csv(args.open_input)
    objective_tols = parse_csv_floats(args.objective_tols)

    h11 = float(stat["h_alpha_alpha_exact"])
    h12 = float(stat["h_alpha_beta_exact"])
    h22 = float(stat["h_beta_beta_exact"])
    h_min = float(min(stat["h_alpha_beta_eig1"], stat["h_alpha_beta_eig2"]))
    c4 = float(stat["lambda_quartic_coeff"])
    H = np.array([[h11, h12], [h12, h22]], dtype=float)

    scan = scan.copy()
    scan["delta_alpha"] = scan["alpha"] - 1.0
    scan["delta_beta"] = scan["beta"] - 1.0
    scan["quad_gap"] = [
        0.5 * float(np.dot([da, db], H @ np.array([da, db], dtype=float)))
        for da, db in zip(scan["delta_alpha"], scan["delta_beta"])
    ]
    scan["quartic_gap"] = c4 * (scan["lambda"] ** 4)
    scan["local_model_gap"] = scan["quad_gap"] + scan["quartic_gap"]

    local_scan = scan[
        (np.abs(scan["delta_alpha"]) <= args.alpha_radius)
        & (np.abs(scan["delta_beta"]) <= args.beta_radius)
        & (np.abs(scan["lambda"]) <= args.lambda_radius)
        & (scan["local_model_gap"] > 0.0)
    ].copy()
    local_scan["coercivity_ratio"] = local_scan["j_parent_statement"] / np.maximum(local_scan["local_model_gap"], 1e-30)
    c_local = float(local_scan["coercivity_ratio"].min())

    alpha_candidates = sorted(local_scan["alpha"].unique())
    beta_candidates = sorted(local_scan["beta"].unique())
    s_part = parent_df["s_part"].to_numpy(dtype=float)
    s_schur = parent_df["s_schur"].to_numpy(dtype=float)

    l_box = np.zeros(len(parent_df), dtype=float)
    for alpha in alpha_candidates:
        for beta in beta_candidates:
            l_box = np.maximum(l_box, rowwise_grad_norm(float(alpha), float(beta), s_part, s_schur))

    parent_df["g_grad_box"] = l_box

    overlap = parent_df.merge(
        open_df[["D", "omega1", "delta_micro", "Szz_0", "Sxx_delta"]],
        on="D",
        how="inner",
    )
    overlap["r_on_shell"] = overlap["delta_micro"] / np.maximum(overlap["omega1"], 1e-30)

    detail_rows = []
    for row in parent_df.itertuples(index=False):
        detail_rows.append(
            {
                "D": float(row.D),
                "eta": float(row.eta),
                "g_proxy_canonical": float(row.g_proxy_canonical),
                "g_grad_box": float(row.g_grad_box),
                "s_part": float(row.s_part),
                "s_schur": float(row.s_schur),
            }
        )
    detail = pd.DataFrame(detail_rows)

    summary = pd.DataFrame(
        [
            {
                "alpha_radius": float(args.alpha_radius),
                "beta_radius": float(args.beta_radius),
                "lambda_radius": float(args.lambda_radius),
                "local_scan_points": int(len(local_scan)),
                "coercivity_ratio_min": c_local,
                "coercivity_ratio_p10": float(local_scan["coercivity_ratio"].quantile(0.10)),
                "coercivity_ratio_p50": float(local_scan["coercivity_ratio"].median()),
                "coercivity_ratio_max": float(local_scan["coercivity_ratio"].max()),
                "quadratic_hessian_eigmin": h_min,
                "lambda_quartic_coeff": c4,
                "g_proxy_canonical_min": float(parent_df["g_proxy_canonical"].min()),
                "g_proxy_canonical_p50": float(parent_df["g_proxy_canonical"].median()),
                "g_proxy_canonical_max": float(parent_df["g_proxy_canonical"].max()),
                "g_grad_box_max": float(parent_df["g_grad_box"].max()),
                "g_grad_box_p50": float(parent_df["g_grad_box"].median()),
            }
        ]
    )

    tol_rows = []
    for tol in objective_tols:
        rho_tol = np.sqrt(2.0 * tol / max(c_local * h_min, 1e-30))
        analytic_valid = rho_tol <= max(args.alpha_radius, args.beta_radius)

        g_floor_analytic = np.maximum(parent_df["g_proxy_canonical"] - parent_df["g_grad_box"] * rho_tol, 0.0)

        admissible = local_scan[local_scan["j_parent_statement"] <= tol].copy()
        if len(admissible):
            rowwise_vals = []
            flat_vals = []
            for cand in admissible.itertuples(index=False):
                g = rowwise_g(float(cand.alpha), float(cand.beta), s_part, s_schur)
                flat_vals.append(g)
                rowwise_vals.append(g)
            exact_matrix = np.stack(rowwise_vals, axis=0)
            exact_rowmin = exact_matrix.min(axis=0)
            exact_flat = np.concatenate(flat_vals)
            exact_family_min = float(exact_flat.min())
            exact_rowmin_min = float(exact_rowmin.min())
            exact_rowmin_p10 = float(pd.Series(exact_rowmin).quantile(0.10))
            exact_rowmin_p50 = float(pd.Series(exact_rowmin).median())
        else:
            exact_family_min = np.nan
            exact_rowmin_min = np.nan
            exact_rowmin_p10 = np.nan
            exact_rowmin_p50 = np.nan

        g_uniform = float(g_floor_analytic.min())
        sep = lorentz_factor((overlap["r_on_shell"] / max(g_uniform, 1e-30)).to_numpy(dtype=float)) - lorentz_factor(
            overlap["r_on_shell"].to_numpy(dtype=float)
        )
        eps_factor = (1.0 + overlap["r_on_shell"] ** 2) / (g_uniform * g_uniform + overlap["r_on_shell"] ** 2)
        sep_min = float(np.min(sep))
        eps_factor_max = float(np.max(eps_factor))
        if args.delta_q_ref < sep_min:
            tail_upper = float(args.delta_q_ref / sep_min)
            eps0_upper = float(args.delta_q_ref / (sep_min - args.delta_q_ref))
            eps_delta_upper = float(eps0_upper * eps_factor_max)
        else:
            tail_upper = np.nan
            eps0_upper = np.nan
            eps_delta_upper = np.nan

        tol_rows.append(
            {
                "objective_tol": float(tol),
                "rho_tol": float(rho_tol),
                "analytic_valid_in_box": int(bool(analytic_valid)),
                "admissible_scan_points": int(len(admissible)),
                "analytic_g_floor_min": g_uniform,
                "analytic_g_floor_p10": float(pd.Series(g_floor_analytic).quantile(0.10)),
                "analytic_g_floor_p50": float(pd.Series(g_floor_analytic).median()),
                "exact_family_g_min": exact_family_min,
                "exact_rowmin_g_min": exact_rowmin_min,
                "exact_rowmin_g_p10": exact_rowmin_p10,
                "exact_rowmin_g_p50": exact_rowmin_p50,
                "separator_min_from_analytic_floor": sep_min,
                "eps_delta_factor_max_from_analytic_floor": eps_factor_max,
                "delta_q_ref": float(args.delta_q_ref),
                "tail_mass_upper_at_delta_q_ref": tail_upper,
                "eps0_upper_at_delta_q_ref": eps0_upper,
                "eps_delta_upper_at_delta_q_ref": eps_delta_upper,
            }
        )
    tol_summary = pd.DataFrame(tol_rows)

    for output in [args.detail_output, args.summary_output, args.tolerance_output]:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    detail.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    tol_summary.to_csv(args.tolerance_output, index=False)

    print(summary.to_string(index=False))
    print()
    print(tol_summary.to_string(index=False))


if __name__ == "__main__":
    main()
