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


def build_gap_conditioned_detail(
    df: pd.DataFrame,
    gap_ratios: list[float],
) -> pd.DataFrame:
    base = df.copy()
    base["r_on_shell"] = base["delta_micro"] / base["omega1"]
    base["q_on_shell"] = base["Sxx_delta"] / base["Szz_0"]
    base["q_single_pole"] = lorentz_factor(base["r_on_shell"].to_numpy(dtype=float))
    base["delta_q_abs"] = np.abs(base["q_on_shell"] - base["q_single_pole"])

    rows: list[dict[str, float]] = []
    for row in base.itertuples(index=False):
        r = float(row.r_on_shell)
        delta_q = float(row.delta_q_abs)
        q_single = float(row.q_single_pole)
        for gap_ratio in gap_ratios:
            q_tail_floor = float(lorentz_factor(np.asarray([r / gap_ratio]))[0])
            sep = q_tail_floor - q_single
            eps_factor = (1.0 + r * r) / (gap_ratio * gap_ratio + r * r)
            actual_tail_mass_upper = delta_q / sep if sep > 0.0 else np.nan
            if sep > delta_q:
                actual_eps0_upper = delta_q / (sep - delta_q)
                actual_eps_delta_upper = actual_eps0_upper * eps_factor
            else:
                actual_eps0_upper = np.nan
                actual_eps_delta_upper = np.nan

            rows.append(
                {
                    "D": float(row.D),
                    "gap_ratio": float(gap_ratio),
                    "r_on_shell": r,
                    "q_on_shell": float(row.q_on_shell),
                    "q_single_pole": q_single,
                    "delta_q_abs": delta_q,
                    "q_tail_floor": q_tail_floor,
                    "gap_separator": sep,
                    "eps_delta_factor": eps_factor,
                    "actual_tail_mass_upper": actual_tail_mass_upper,
                    "actual_eps0_upper": actual_eps0_upper,
                    "actual_eps_delta_upper": actual_eps_delta_upper,
                }
            )
    return pd.DataFrame(rows)


def build_tolerance_grid(
    detail: pd.DataFrame,
    delta_q_tolerances: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in detail.itertuples(index=False):
        sep = float(row.gap_separator)
        for tol in delta_q_tolerances:
            if tol < sep:
                tail_mass_upper = tol / sep
                eps0_upper = tol / (sep - tol)
                eps_delta_upper = eps0_upper * float(row.eps_delta_factor)
            else:
                tail_mass_upper = np.nan
                eps0_upper = np.nan
                eps_delta_upper = np.nan
            rows.append(
                {
                    "D": float(row.D),
                    "gap_ratio": float(row.gap_ratio),
                    "delta_q_tol": float(tol),
                    "gap_separator": sep,
                    "eps_delta_factor": float(row.eps_delta_factor),
                    "tail_mass_upper": tail_mass_upper,
                    "eps0_upper": eps0_upper,
                    "eps_delta_upper": eps_delta_upper,
                }
            )
    return pd.DataFrame(rows)


def summarize_detail(detail: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail.groupby("gap_ratio", as_index=False)
        .agg(
            D_min=("D", "min"),
            D_max=("D", "max"),
            r_min=("r_on_shell", "min"),
            r_max=("r_on_shell", "max"),
            gap_separator_min=("gap_separator", "min"),
            gap_separator_max=("gap_separator", "max"),
            eps_delta_factor_min=("eps_delta_factor", "min"),
            eps_delta_factor_max=("eps_delta_factor", "max"),
            max_delta_q_abs=("delta_q_abs", "max"),
            actual_tail_mass_upper_max=("actual_tail_mass_upper", "max"),
            actual_eps0_upper_max=("actual_eps0_upper", "max"),
            actual_eps_delta_upper_max=("actual_eps_delta_upper", "max"),
        )
    )
    return summary


def summarize_tolerances(tolerance_grid: pd.DataFrame) -> pd.DataFrame:
    summary = (
        tolerance_grid.groupby(["gap_ratio", "delta_q_tol"], as_index=False)
        .agg(
            D_min=("D", "min"),
            D_max=("D", "max"),
            gap_separator_min=("gap_separator", "min"),
            gap_separator_max=("gap_separator", "max"),
            tail_mass_upper_max=("tail_mass_upper", "max"),
            eps0_upper_max=("eps0_upper", "max"),
            eps_delta_upper_max=("eps_delta_upper", "max"),
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the gap-conditioned first-mode remainder bound for the "
            "open-system Schur-resolvent theorem target."
        )
    )
    parser.add_argument(
        "--input",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_micro_bridge_map.csv"
        ),
    )
    parser.add_argument(
        "--gap-ratios",
        default="1.25,1.5,2.0,3.0",
        help="Comma-separated candidate lower bounds for Omega2 / Omega1.",
    )
    parser.add_argument(
        "--delta-q-tols",
        default="1e-3,5e-3,1e-2",
        help="Comma-separated hypothetical on-shell ratio defects.",
    )
    parser.add_argument(
        "--detail-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_first_mode_gap_bound_detail.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_first_mode_gap_bound_summary.csv"
        ),
    )
    parser.add_argument(
        "--tolerance-output",
        default=(
            "/Users/boypatrick/codex/PSLT_quantam/output/chi_open_system/"
            "chi_open_system_first_mode_gap_bound_tolerance_summary.csv"
        ),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    gap_ratios = parse_csv_floats(args.gap_ratios)
    delta_q_tolerances = parse_csv_floats(args.delta_q_tols)

    detail = build_gap_conditioned_detail(df, gap_ratios)
    tolerance_grid = build_tolerance_grid(detail, delta_q_tolerances)
    summary = summarize_detail(detail)
    tolerance_summary = summarize_tolerances(tolerance_grid)

    for output in [args.detail_output, args.summary_output, args.tolerance_output]:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    detail.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    tolerance_summary.to_csv(args.tolerance_output, index=False)

    print(summary.to_string(index=False))
    print()
    print(tolerance_summary.to_string(index=False))


if __name__ == "__main__":
    main()
