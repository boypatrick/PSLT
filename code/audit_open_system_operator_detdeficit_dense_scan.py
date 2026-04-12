#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def q(v: np.ndarray | pd.Series, p: float) -> float:
    return float(pd.Series(np.asarray(v, dtype=float)).quantile(p))


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


def build_grid(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(round((hi - lo) / step))
    return np.array([lo + i * step for i in range(n + 1)], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Densify the exact projected parent-kernel local scan and audit the determinant deficit "
            "u_a=lambda^2 xi_a(alpha,beta) directly on admissible family points."
        )
    )
    parser.add_argument(
        "--parented-map-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/paper/"
            "hll_uv_to_eft_map_uv_action_loop_eymh_parented_D21E21_fix.csv"
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
    parser.add_argument("--objective-tols", default="1.500329e-7,5e-7,1e-6,2e-6,2.5e-6")
    parser.add_argument("--alpha-min", type=float, default=0.95)
    parser.add_argument("--alpha-max", type=float, default=1.05)
    parser.add_argument("--alpha-step", type=float, default=0.005)
    parser.add_argument("--beta-min", type=float, default=0.95)
    parser.add_argument("--beta-max", type=float, default=1.05)
    parser.add_argument("--beta-step", type=float, default=0.005)
    parser.add_argument("--lambda-min", type=float, default=-0.20)
    parser.add_argument("--lambda-max", type=float, default=0.20)
    parser.add_argument("--lambda-step", type=float, default=0.005)
    parser.add_argument("--delta-q-ref", type=float, default=1e-3)
    parser.add_argument(
        "--detail-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_detdeficit_dense_scan_detail.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_detdeficit_dense_scan_summary.csv"
        ),
    )
    args = parser.parse_args()

    parented = pd.read_csv(args.parented_map_input)
    open_df = pd.read_csv(args.open_input)
    bound_df = pd.read_csv(args.operator_bound_input)

    overlap = parented.merge(open_df[["D", "omega1", "delta_micro"]], on="D", how="inner")
    coeff_l1 = np.maximum(np.asarray(overlap["coeff_l1"], dtype=float), 1e-30)
    coeff_l2 = np.maximum(np.asarray(overlap["coeff_l2"], dtype=float), 1e-30)
    k_part = coeff_l1 / coeff_l2
    s_part = np.log(k_part)

    chi_tree = np.maximum(np.asarray(overlap["tree_diag_susceptibility_parented"], dtype=float), 0.0)
    g_schur = 1.0 + chi_tree
    s_schur = np.log(g_schur)
    a_ref = np.asarray(overlap["coeff_participation_access_parented"], dtype=float) * np.asarray(
        overlap["tree_diag_compressibility_parented"], dtype=float
    )
    r_on_shell = (overlap["delta_micro"] / np.maximum(overlap["omega1"], 1e-30)).to_numpy(dtype=float)

    alphas = build_grid(args.alpha_min, args.alpha_max, args.alpha_step)
    betas = build_grid(args.beta_min, args.beta_max, args.beta_step)
    lambdas = build_grid(args.lambda_min, args.lambda_max, args.lambda_step)
    objective_tols = parse_csv_floats(args.objective_tols)

    rows: list[dict[str, float]] = []
    for alpha in alphas:
        k11 = np.exp(alpha * s_part)
        for beta in betas:
            k22 = np.exp(beta * s_schur)
            xi_ab = np.maximum((k11 - 1.0) * (k22 - 1.0), 0.0) / np.maximum(k11 * k22, 1e-30)
            delta_linear = (alpha - 1.0) * s_part + (beta - 1.0) * s_schur
            for lam in lambdas:
                delta_mix = np.log(np.maximum(1.0 - (lam * lam) * xi_ab, 1e-30))
                delta_total = delta_linear + delta_mix
                j_parent = float(np.mean((a_ref * (np.exp(-0.5 * delta_total) - 1.0)) ** 2))
                u = (lam * lam) * xi_ab
                rows.append(
                    {
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "lambda": float(lam),
                        "j_parent_statement": j_parent,
                        "u_overlap_max": float(np.max(u)),
                        "u_overlap_p95": q(u, 0.95),
                        "u_overlap_p50": q(u, 0.50),
                        "xi_overlap_max": float(np.max(xi_ab)),
                        "xi_overlap_p95": q(xi_ab, 0.95),
                    }
                )

    detail = pd.DataFrame(rows).sort_values(["j_parent_statement", "u_overlap_max", "alpha", "beta", "lambda"])

    summary_rows = []
    for tol in objective_tols:
        family_g_floor = float(bound_df.iloc[(bound_df["objective_tol"] - tol).abs().idxmin()]["analytic_g_floor_min"])
        admissible = detail[detail["j_parent_statement"] <= tol].copy()
        if admissible.empty:
            continue
        best = admissible.sort_values(["u_overlap_max", "j_parent_statement"], ascending=[False, True]).iloc[0]
        u_upper = float(best["u_overlap_max"])
        g_lifted = float(np.sqrt(max(1.0 - u_upper, 0.0)) * family_g_floor)
        evals = evaluate_gap_floor(g_lifted, r_on_shell, args.delta_q_ref)
        summary_rows.append(
            {
                "objective_tol": float(tol),
                "admissible_dense_points": int(len(admissible)),
                "family_g_floor": family_g_floor,
                "u_dense_upper": u_upper,
                "trA_upper": tr_upper_from_u(u_upper),
                "opA_upper": u_upper,
                "heff_loewner_factor": float(1.0 - u_upper),
                "cor1_logdet_gap_upper": cor1_gap_upper_from_u(u_upper),
                "lifted_g_floor": g_lifted,
                "separator_min": evals["separator_min"],
                "eps_delta_upper": evals["eps_delta_upper"],
                "alpha_at_u_max": float(best["alpha"]),
                "beta_at_u_max": float(best["beta"]),
                "lambda_at_u_max": float(best["lambda"]),
                "j_parent_at_u_max": float(best["j_parent_statement"]),
                "u_p95_among_admissible": q(admissible["u_overlap_max"], 0.95),
                "u_p50_among_admissible": q(admissible["u_overlap_max"], 0.50),
                "delta_q_ref": float(args.delta_q_ref),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("objective_tol")

    for out in [args.detail_output, args.summary_output]:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print(args.detail_output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
