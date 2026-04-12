#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def closest_row(df: pd.DataFrame, column: str, target: float) -> pd.Series:
    idx = (df[column] - target).abs().idxmin()
    return df.loc[idx]


def tr_upper_from_u(u: float) -> float:
    return float(2.0 * (1.0 - np.sqrt(max(1.0 - u, 0.0))))


def cor1_gap_upper_from_u(u: float) -> float:
    return float(2.0 * u / max(1.0 - u, 1e-30))


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a continuum-style neighborhood bound on the determinant deficit "
            "u_a=lambda^2 xi_a(alpha,beta) using the local coercive budget "
            "0.5 dtheta^T H dtheta + C4 lambda^4 <= J/c_loc."
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
        "--open-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_micro_bridge_map.csv"
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
        "--operator-bound-summary-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_gap_lower_bound_summary.csv"
        ),
    )
    parser.add_argument(
        "--operator-bound-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_gap_lower_bound_tolerance_summary.csv"
        ),
    )
    parser.add_argument("--objective-tols", default="1.500329e-7,5e-7,1e-6,2e-6,2.5e-6")
    parser.add_argument("--alpha-radius", type=float, default=0.1)
    parser.add_argument("--beta-radius", type=float, default=0.1)
    parser.add_argument("--lambda-radius", type=float, default=0.2)
    parser.add_argument("--alpha-points", type=int, default=101)
    parser.add_argument("--beta-points", type=int, default=101)
    parser.add_argument("--delta-q-ref", type=float, default=1e-3)
    parser.add_argument(
        "--summary-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_detdeficit_neighborhood_bound_summary.csv"
        ),
    )
    args = parser.parse_args()

    parent_df = pd.read_csv(args.parent_blockdet_input)
    parent_df = parent_df[parent_df["parent_blockdet_action"] > 1e-12].copy()
    open_df = pd.read_csv(args.open_input)
    stat = pd.read_csv(args.stationarity_input).iloc[0]
    bound_summary = pd.read_csv(args.operator_bound_summary_input).iloc[0]
    bound_df = pd.read_csv(args.operator_bound_input)

    overlap = parent_df.merge(open_df[["D", "omega1", "delta_micro"]], on="D", how="inner")
    s_part = np.log(overlap["parent_blockdet_k11"].to_numpy(dtype=float))
    s_schur = np.log(overlap["parent_blockdet_schur_norm"].to_numpy(dtype=float))
    r_on_shell = (overlap["delta_micro"] / np.maximum(overlap["omega1"], 1e-30)).to_numpy(dtype=float)

    c_loc = float(bound_summary["coercivity_ratio_min"])
    c4 = float(stat["lambda_quartic_coeff"])
    H = np.array(
        [
            [float(stat["h_alpha_alpha_exact"]), float(stat["h_alpha_beta_exact"])],
            [float(stat["h_alpha_beta_exact"]), float(stat["h_beta_beta_exact"])],
        ],
        dtype=float,
    )

    alpha_grid = np.linspace(1.0 - args.alpha_radius, 1.0 + args.alpha_radius, args.alpha_points)
    beta_grid = np.linspace(1.0 - args.beta_radius, 1.0 + args.beta_radius, args.beta_points)
    objective_tols = parse_csv_floats(args.objective_tols)

    rows = []
    for tol in objective_tols:
        family_g_floor = float(closest_row(bound_df, "objective_tol", tol)["analytic_g_floor_min"])
        budget = float(tol / max(c_loc, 1e-30))

        best_u = -1.0
        best = None
        feasible_points = 0
        for alpha in alpha_grid:
            da = alpha - 1.0
            for beta in beta_grid:
                db = beta - 1.0
                quad = 0.5 * float(np.dot([da, db], H @ np.array([da, db], dtype=float)))
                if quad > budget:
                    continue
                feasible_points += 1
                lam2_budget = float(np.sqrt(max((budget - quad) / max(c4, 1e-30), 0.0)))
                lam2_upper = min(lam2_budget, args.lambda_radius * args.lambda_radius)
                k11 = np.exp(alpha * s_part)
                k22 = np.exp(beta * s_schur)
                xi_ab = np.maximum((k11 - 1.0) * (k22 - 1.0), 0.0) / np.maximum(k11 * k22, 1e-30)
                xi_upper = float(np.max(xi_ab))
                u = lam2_upper * xi_upper
                if u > best_u:
                    best_u = u
                    best = {
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "quad_budget_used": quad,
                        "lambda2_upper": lam2_upper,
                        "lambda_upper": float(np.sqrt(max(lam2_upper, 0.0))),
                        "xi_upper": xi_upper,
                    }

        assert best is not None
        g_lifted = float(np.sqrt(max(1.0 - best_u, 0.0)) * family_g_floor)
        evals = evaluate_gap_floor(g_lifted, r_on_shell, args.delta_q_ref)
        rows.append(
            {
                "objective_tol": float(tol),
                "feasible_grid_points": int(feasible_points),
                "family_g_floor": family_g_floor,
                "u_neighborhood_upper": float(best_u),
                "trA_upper": tr_upper_from_u(float(best_u)),
                "opA_upper": float(best_u),
                "heff_loewner_factor": float(1.0 - best_u),
                "cor1_logdet_gap_upper": cor1_gap_upper_from_u(float(best_u)),
                "lifted_g_floor": g_lifted,
                "separator_min": evals["separator_min"],
                "eps_delta_upper": evals["eps_delta_upper"],
                "alpha_at_u_max": best["alpha"],
                "beta_at_u_max": best["beta"],
                "lambda_upper_at_u_max": best["lambda_upper"],
                "lambda2_upper_at_u_max": best["lambda2_upper"],
                "quad_budget_used_at_u_max": best["quad_budget_used"],
                "xi_upper_at_u_max": best["xi_upper"],
                "s_part_min": float(s_part.min()),
                "s_schur_min": float(s_schur.min()),
                "delta_q_ref": float(args.delta_q_ref),
            }
        )

    summary = pd.DataFrame(rows).sort_values("objective_tol")
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    print(args.summary_output)


if __name__ == "__main__":
    main()
