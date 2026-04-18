#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the on-shell ratio defect delta_q from the parent-bath statement. "
            "This checks that the normalized bath anisotropy and the gap-locked Lorentzian "
            "response give the same q_a = R(DeltaE)/R(0) on the audited knot set."
        )
    )
    parser.add_argument(
        "--statement-map-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_parent_bath_statement_map.csv"
        ),
    )
    parser.add_argument(
        "--micro-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_micro_bridge_map.csv"
        ),
    )
    parser.add_argument(
        "--gap-floor-input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_operator_gap_lower_bound_tolerance_summary.csv"
        ),
    )
    parser.add_argument("--objective-tol", type=float, default=1e-6)
    parser.add_argument(
        "--detail-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_deltaq_source_detail.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_deltaq_source_summary.csv"
        ),
    )
    args = parser.parse_args()

    statement = pd.read_csv(args.statement_map_input)
    micro = pd.read_csv(args.micro_input)
    gap_floor = pd.read_csv(args.gap_floor_input)

    cols_statement = [
        "D",
        "bath_shape_phi",
        "bath_shape_mix",
        "parent_log_bath_anisotropy",
        "parent_log_rate_anisotropy",
        "parent_log_sys_anisotropy",
    ]
    cols_micro = [
        "D",
        "delta_micro",
        "omega1",
        "tau_env",
        "Szz_0",
        "Sxx_delta",
    ]
    df = statement[cols_statement].merge(micro[cols_micro], on="D", how="inner").sort_values("D").reset_index(drop=True)

    q_parent_log = np.exp(df["parent_log_bath_anisotropy"].to_numpy(dtype=float))
    q_parent_shape = (
        df["bath_shape_mix"].to_numpy(dtype=float) / np.maximum(df["bath_shape_phi"].to_numpy(dtype=float), 1e-30)
    )
    q_micro = df["Sxx_delta"].to_numpy(dtype=float) / np.maximum(df["Szz_0"].to_numpy(dtype=float), 1e-30)
    r_on_shell = df["delta_micro"].to_numpy(dtype=float) / np.maximum(df["omega1"].to_numpy(dtype=float), 1e-30)
    q_gap_locked = lorentz_factor(r_on_shell)
    tau_gap_locked = 1.0 / np.maximum(df["omega1"].to_numpy(dtype=float), 1e-30)

    df["q_parent_log"] = q_parent_log
    df["q_parent_shape"] = q_parent_shape
    df["q_micro"] = q_micro
    df["q_gap_locked"] = q_gap_locked
    df["tau_gap_locked"] = tau_gap_locked
    df["delta_q_parent_log_vs_gap"] = q_parent_log - q_gap_locked
    df["delta_q_parent_shape_vs_gap"] = q_parent_shape - q_gap_locked
    df["delta_q_micro_vs_gap"] = q_micro - q_gap_locked
    df["delta_q_parent_log_vs_micro"] = q_parent_log - q_micro
    df["delta_q_parent_shape_vs_micro"] = q_parent_shape - q_micro
    df["tau_gap_lock_residual_abs"] = np.abs(df["tau_env"] - df["tau_gap_locked"])
    df["q_parent_log_vs_shape_residual_abs"] = np.abs(df["q_parent_log"] - df["q_parent_shape"])
    df["q_parent_log_vs_gap_residual_abs"] = np.abs(df["delta_q_parent_log_vs_gap"])
    df["q_micro_vs_gap_residual_abs"] = np.abs(df["delta_q_micro_vs_gap"])

    gap_row = closest_row(gap_floor, "objective_tol", args.objective_tol)
    analytic_g_floor = float(gap_row["analytic_g_floor_min"])
    exact_g_floor = float(gap_row["exact_family_g_min"])
    delta_q_actual = float(df["q_parent_log_vs_gap_residual_abs"].max())

    analytic_eval = evaluate_gap_floor(analytic_g_floor, r_on_shell, delta_q_actual)
    exact_eval = evaluate_gap_floor(exact_g_floor, r_on_shell, delta_q_actual)

    summary = pd.DataFrame(
        [
            {
                "n_rows": int(len(df)),
                "D_min": float(df["D"].min()),
                "D_max": float(df["D"].max()),
                "r_on_shell_min": float(np.min(r_on_shell)),
                "r_on_shell_max": float(np.max(r_on_shell)),
                "max_abs_tau_gap_lock_residual": float(df["tau_gap_lock_residual_abs"].max()),
                "max_abs_q_parent_log_vs_shape_residual": float(df["q_parent_log_vs_shape_residual_abs"].max()),
                "max_abs_delta_q_parent_log_vs_gap": float(df["q_parent_log_vs_gap_residual_abs"].max()),
                "max_abs_delta_q_parent_shape_vs_gap": float(np.abs(df["delta_q_parent_shape_vs_gap"]).max()),
                "max_abs_delta_q_micro_vs_gap": float(df["q_micro_vs_gap_residual_abs"].max()),
                "max_abs_delta_q_parent_log_vs_micro": float(np.abs(df["delta_q_parent_log_vs_micro"]).max()),
                "analytic_g_floor": analytic_g_floor,
                "exact_g_floor": exact_g_floor,
                "analytic_separator_min": analytic_eval["separator_min"],
                "analytic_tail_mass_upper_from_actual_deltaq": analytic_eval["tail_mass_upper"],
                "analytic_eps_delta_upper_from_actual_deltaq": analytic_eval["eps_delta_upper"],
                "exact_separator_min": exact_eval["separator_min"],
                "exact_tail_mass_upper_from_actual_deltaq": exact_eval["tail_mass_upper"],
                "exact_eps_delta_upper_from_actual_deltaq": exact_eval["eps_delta_upper"],
            }
        ]
    )

    Path(args.detail_output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print(args.detail_output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
