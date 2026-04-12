#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def q(v: np.ndarray | pd.Series, p: float) -> float:
    return float(pd.Series(np.asarray(v, dtype=float)).quantile(p))


def parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def closest_row(df: pd.DataFrame, column: str, target: float) -> pd.Series:
    idx = (df[column] - target).abs().idxmin()
    return df.loc[idx]


def lorentz_factor(r: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + r * r)


def evaluate_gap_floor(g_floor: float, r_on_shell: np.ndarray, delta_q_ref: float) -> dict[str, float]:
    sep = lorentz_factor(r_on_shell / max(g_floor, 1e-30)) - lorentz_factor(r_on_shell)
    sep_min = float(np.min(sep))
    eps_factor_max = float(np.max((1.0 + r_on_shell * r_on_shell) / (g_floor * g_floor + r_on_shell * r_on_shell)))
    if delta_q_ref < sep_min:
        tail_upper = float(delta_q_ref / sep_min)
        eps0_upper = float(delta_q_ref / (sep_min - delta_q_ref))
        eps_delta_upper = float(eps0_upper * eps_factor_max)
    else:
        tail_upper = np.nan
        eps0_upper = np.nan
        eps_delta_upper = np.nan
    return {
        "separator_min": sep_min,
        "eps_delta_factor_max": eps_factor_max,
        "tail_mass_upper": tail_upper,
        "eps0_upper": eps0_upper,
        "eps_delta_upper": eps_delta_upper,
    }


def tr_upper_from_u(u: float) -> float:
    return float(2.0 * (1.0 - np.sqrt(max(1.0 - u, 0.0))))


def cor1_gap_upper_from_u(u: float) -> float:
    return float(2.0 * u / max(1.0 - u, 1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit parent-side determinant-deficit sources u_a = 1-det(I-A_a) directly on the exact "
            "projected kernel family, and compare them with the conservative raw off-diagonal envelope."
        )
    )
    parser.add_argument(
        "--parent-scan-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_action_loop_eymh_parent_kernel_statement_scan_D21E21_fix.csv"
        ),
    )
    parser.add_argument(
        "--parent-blockdet-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_action_loop_eymh_parent_blockdet_audit_map_D21E21_fix.csv"
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
        "--operator-bound-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_gap_lower_bound_tolerance_summary.csv"
        ),
    )
    parser.add_argument(
        "--offdiag-proxy-summary-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_sigma_offdiag_proxy_summary.csv"
        ),
    )
    parser.add_argument("--delta-q-ref", type=float, default=1e-3)
    parser.add_argument(
        "--objective-tols",
        default="1.500329e-7,5e-7,1e-6,2e-6,2.5e-6",
    )
    parser.add_argument("--alpha-radius", type=float, default=0.1)
    parser.add_argument("--beta-radius", type=float, default=0.1)
    parser.add_argument("--lambda-radius", type=float, default=0.2)
    parser.add_argument(
        "--detail-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_detdeficit_source_detail.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_detdeficit_source_summary.csv"
        ),
    )
    args = parser.parse_args()

    scan = pd.read_csv(args.parent_scan_input)
    parent_df = pd.read_csv(args.parent_blockdet_input)
    parent_df = parent_df[parent_df["parent_blockdet_action"] > 1e-12].copy()
    open_df = pd.read_csv(args.open_input)
    bound_df = pd.read_csv(args.operator_bound_input)
    proxy_summary = pd.read_csv(args.offdiag_proxy_summary_input)

    parent_df["s_part"] = np.log(parent_df["parent_blockdet_k11"])
    parent_df["s_schur"] = np.log(parent_df["parent_blockdet_schur_norm"])
    overlap = parent_df.merge(open_df[["D", "omega1", "delta_micro"]], on="D", how="inner")
    s_part = overlap["s_part"].to_numpy(dtype=float)
    s_schur = overlap["s_schur"].to_numpy(dtype=float)
    r_on_shell = (overlap["delta_micro"] / np.maximum(overlap["omega1"], 1e-30)).to_numpy(dtype=float)

    raw_row = proxy_summary[
        proxy_summary["proxy_name"].eq("runnerup_blend_abs_action")
    ].sort_values("overlap_p95_abs_residual").iloc[0]
    xi_raw_p95 = float(raw_row["xi_upper_p95_envelope"])

    objective_tols = parse_csv_floats(args.objective_tols)

    local_scan = scan[
        (np.abs(scan["alpha"] - 1.0) <= args.alpha_radius)
        & (np.abs(scan["beta"] - 1.0) <= args.beta_radius)
        & (np.abs(scan["lambda"]) <= args.lambda_radius)
    ].copy()

    detail_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, float]] = []

    for tol in objective_tols:
        bound_row = closest_row(bound_df, "objective_tol", tol)
        family_g_floor = float(bound_row["analytic_g_floor_min"])

        admissible = local_scan[local_scan["j_parent_statement"] <= tol].copy()
        if admissible.empty:
            continue

        best_u_max = -1.0
        best_summary: dict[str, float] | None = None

        for _, row in admissible.iterrows():
            alpha = float(row["alpha"])
            beta = float(row["beta"])
            lam = float(row["lambda"])

            k11 = np.exp(alpha * s_part)
            k22 = np.exp(beta * s_schur)
            xi_ab = np.maximum((k11 - 1.0) * (k22 - 1.0), 0.0) / np.maximum(k11 * k22, 1e-30)
            u = (lam * lam) * xi_ab
            u_max = float(np.max(u))
            u_p95 = q(u, 0.95)
            u_p50 = q(u, 0.50)
            xi_max = float(np.max(xi_ab))
            xi_p95 = q(xi_ab, 0.95)

            op_upper = u_max
            tr_upper = tr_upper_from_u(u_max)
            heff_factor = float(1.0 - u_max)
            cor1_upper = cor1_gap_upper_from_u(u_max)
            g_lifted = float(np.sqrt(max(1.0 - u_max, 0.0)) * family_g_floor)
            evals = evaluate_gap_floor(g_lifted, r_on_shell, args.delta_q_ref)

            row_record = {
                "objective_tol": float(tol),
                "family_g_floor": family_g_floor,
                "alpha": alpha,
                "beta": beta,
                "lambda": lam,
                "j_parent_statement": float(row["j_parent_statement"]),
                "u_exact_overlap_max": u_max,
                "u_exact_overlap_p95": u_p95,
                "u_exact_overlap_p50": u_p50,
                "xi_exact_overlap_max": xi_max,
                "xi_exact_overlap_p95": xi_p95,
                "opA_upper_from_u_exact": op_upper,
                "trA_upper_from_u_exact": tr_upper,
                "heff_loewner_factor_from_u_exact": heff_factor,
                "cor1_logdet_gap_upper_from_u_exact": cor1_upper,
                "lifted_g_floor_from_u_exact": g_lifted,
                "separator_min_from_u_exact": evals["separator_min"],
                "eps_delta_upper_from_u_exact": evals["eps_delta_upper"],
                "delta_q_ref": float(args.delta_q_ref),
            }
            detail_rows.append(row_record)

            if u_max > best_u_max:
                best_u_max = u_max
                best_summary = row_record

        lam_abs_max = float(np.max(np.abs(admissible["lambda"])))
        u_raw_env = float((lam_abs_max * lam_abs_max) * xi_raw_p95)
        raw_heff = float(1.0 - u_raw_env)
        raw_g_lifted = float(np.sqrt(max(1.0 - u_raw_env, 0.0)) * family_g_floor)
        raw_evals = evaluate_gap_floor(raw_g_lifted, r_on_shell, args.delta_q_ref)

        assert best_summary is not None
        summary_rows.extend(
            [
                {
                    "objective_tol": float(tol),
                    "source_type": "exact admissible family",
                    "admissible_scan_points": int(len(admissible)),
                    "lambda_abs_max_admissible": lam_abs_max,
                    "family_g_floor": family_g_floor,
                    "u_source_upper": float(best_summary["u_exact_overlap_max"]),
                    "trA_upper": tr_upper_from_u(float(best_summary["u_exact_overlap_max"])),
                    "opA_upper": float(best_summary["u_exact_overlap_max"]),
                    "heff_loewner_factor": float(best_summary["heff_loewner_factor_from_u_exact"]),
                    "cor1_logdet_gap_upper": float(best_summary["cor1_logdet_gap_upper_from_u_exact"]),
                    "lifted_g_floor": float(best_summary["lifted_g_floor_from_u_exact"]),
                    "separator_min": float(best_summary["separator_min_from_u_exact"]),
                    "eps_delta_upper": float(best_summary["eps_delta_upper_from_u_exact"]),
                    "alpha_at_u_max": float(best_summary["alpha"]),
                    "beta_at_u_max": float(best_summary["beta"]),
                    "lambda_at_u_max": float(best_summary["lambda"]),
                    "j_parent_at_u_max": float(best_summary["j_parent_statement"]),
                    "delta_q_ref": float(args.delta_q_ref),
                },
                {
                    "objective_tol": float(tol),
                    "source_type": "raw offdiag envelope",
                    "admissible_scan_points": int(len(admissible)),
                    "lambda_abs_max_admissible": lam_abs_max,
                    "family_g_floor": family_g_floor,
                    "u_source_upper": u_raw_env,
                    "trA_upper": tr_upper_from_u(u_raw_env),
                    "opA_upper": u_raw_env,
                    "heff_loewner_factor": raw_heff,
                    "cor1_logdet_gap_upper": cor1_gap_upper_from_u(u_raw_env),
                    "lifted_g_floor": raw_g_lifted,
                    "separator_min": raw_evals["separator_min"],
                    "eps_delta_upper": raw_evals["eps_delta_upper"],
                    "alpha_at_u_max": np.nan,
                    "beta_at_u_max": np.nan,
                    "lambda_at_u_max": lam_abs_max,
                    "j_parent_at_u_max": np.nan,
                    "delta_q_ref": float(args.delta_q_ref),
                },
            ]
        )

    detail = pd.DataFrame(detail_rows).sort_values(["objective_tol", "u_exact_overlap_max"])
    summary = pd.DataFrame(summary_rows).sort_values(["objective_tol", "source_type"])

    for out in [args.detail_output, args.summary_output]:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print(args.detail_output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
