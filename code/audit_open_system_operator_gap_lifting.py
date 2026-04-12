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


def closest_row(df: pd.DataFrame, column: str, target: float) -> pd.Series:
    idx = (df[column] - target).abs().idxmin()
    return df.loc[idx]


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
        "g_floor": float(g_floor),
        "separator_min": sep_min,
        "eps_delta_factor_max": eps_factor_max,
        "tail_mass_upper": tail_upper,
        "eps0_upper": eps0_upper,
        "eps_delta_upper": eps_delta_upper,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Lift the local operator-family gap floor to a conditional Schur-resolvent "
            "floor using the complement log-det loss from Corollary 1."
        )
    )
    parser.add_argument(
        "--operator-bound-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_gap_lower_bound_tolerance_summary.csv"
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
        "--base-objective-tol",
        type=float,
        default=1e-6,
        help="Local operator-family tolerance whose gap floor is used as the lift base.",
    )
    parser.add_argument(
        "--comp-logdet-tols",
        default="1.500329e-7,1e-6,1e-4,1e-3,1e-2,5e-2",
        help="Comma-separated complement log-det losses sigma=-log det(I-A).",
    )
    parser.add_argument(
        "--delta-q-ref",
        type=float,
        default=1e-3,
        help="Reference on-shell ratio defect used to translate the lifted gap into a remainder estimate.",
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_gap_lifting_summary.csv"
        ),
    )
    parser.add_argument(
        "--lifting-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_gap_lifting_tolerance_summary.csv"
        ),
    )
    args = parser.parse_args()

    bound_df = pd.read_csv(args.operator_bound_input)
    bound_row = closest_row(bound_df, "objective_tol", args.base_objective_tol)
    comp_tols = parse_csv_floats(args.comp_logdet_tols)

    parent_df = pd.read_csv(args.parent_blockdet_input)
    parent_df = parent_df[parent_df["parent_blockdet_action"] > 1e-12].copy()
    open_df = pd.read_csv(args.open_input)
    overlap = parent_df.merge(open_df[["D", "omega1", "delta_micro"]], on="D", how="inner")
    r_on_shell = (overlap["delta_micro"] / np.maximum(overlap["omega1"], 1e-30)).to_numpy(dtype=float)

    analytic_base = float(bound_row["analytic_g_floor_min"])
    exact_base = float(bound_row["exact_family_g_min"])

    summary = pd.DataFrame(
        [
            {
                "base_objective_tol": float(bound_row["objective_tol"]),
                "base_rho_tol": float(bound_row["rho_tol"]),
                "analytic_base_g_floor": analytic_base,
                "exact_base_g_floor": exact_base,
                "delta_q_ref": float(args.delta_q_ref),
                "overlap_rows": int(len(overlap)),
                "overlap_D_min": float(overlap["D"].min()),
                "overlap_D_max": float(overlap["D"].max()),
                "r_on_shell_min": float(r_on_shell.min()),
                "r_on_shell_max": float(r_on_shell.max()),
            }
        ]
    )

    rows = []
    for sigma in comp_tols:
        shrink = float(np.exp(-0.5 * sigma))
        a_norm_upper = float(1.0 - np.exp(-sigma))
        analytic_eval = evaluate_gap_floor(shrink * analytic_base, r_on_shell, args.delta_q_ref)
        exact_eval = evaluate_gap_floor(shrink * exact_base, r_on_shell, args.delta_q_ref)
        rows.append(
            {
                "complement_logdet_tol": float(sigma),
                "a_norm_upper": a_norm_upper,
                "gap_shrink_factor": shrink,
                "analytic_lifted_g_floor": analytic_eval["g_floor"],
                "analytic_separator_min": analytic_eval["separator_min"],
                "analytic_eps_delta_factor_max": analytic_eval["eps_delta_factor_max"],
                "analytic_tail_mass_upper": analytic_eval["tail_mass_upper"],
                "analytic_eps0_upper": analytic_eval["eps0_upper"],
                "analytic_eps_delta_upper": analytic_eval["eps_delta_upper"],
                "exact_lifted_g_floor": exact_eval["g_floor"],
                "exact_separator_min": exact_eval["separator_min"],
                "exact_eps_delta_factor_max": exact_eval["eps_delta_factor_max"],
                "exact_tail_mass_upper": exact_eval["tail_mass_upper"],
                "exact_eps0_upper": exact_eval["eps0_upper"],
                "exact_eps_delta_upper": exact_eval["eps_delta_upper"],
                "delta_q_ref": float(args.delta_q_ref),
            }
        )

    lifting = pd.DataFrame(rows)

    for out in [args.summary_output, args.lifting_output]:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    lifting.to_csv(args.lifting_output, index=False)
    print(args.summary_output)
    print(args.lifting_output)


if __name__ == "__main__":
    main()
