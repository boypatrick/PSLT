#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--window-summary-csv",
        default=str(OUTDIR / "chi_open_system_exact_total_core_window_split_source_summary.csv"),
    )
    ap.add_argument(
        "--threshold-summary-csv",
        default=str(OUTDIR / "chi_open_system_exact_schur_lift_parity_threshold_source_summary.csv"),
    )
    args = ap.parse_args()

    window = pd.read_csv(args.window_summary_csv).iloc[0]
    threshold = pd.read_csv(args.threshold_summary_csv).iloc[0]

    lambda_L = float(window["best_left_lambda_min"])
    core_L = float(window["best_left_core_floor_abs"])
    floor_L = float(window["best_left_induced_floor"])
    eta_L = 1.0 - lambda_L

    lambda_R = float(window["best_right_lambda_min"])
    core_R = float(window["best_right_core_floor_abs"])
    floor_R = float(window["best_right_induced_floor"])
    eta_R = 1.0 - lambda_R

    required_global = float(threshold["required_parity_floor_global"])
    required_local = float(threshold["required_parity_floor_local"])

    overall = min(floor_L, floor_R)
    left_bottleneck = int(floor_L <= floor_R)

    summary = pd.DataFrame(
        [
            {
                "R_star": float(window["R_star"]),
                "cut_D": float(window["best_cut_D"]),
                "left_window": str(window["best_left_window"]),
                "right_window": str(window["best_right_window"]),
                "eta_left_max": eta_L,
                "lambda_left_min": lambda_L,
                "core_left_floor_abs": core_L,
                "induced_left_floor": floor_L,
                "eta_right_max": eta_R,
                "lambda_right_min": lambda_R,
                "core_right_floor_abs": core_R,
                "induced_right_floor": floor_R,
                "overall_piecewise_floor": overall,
                "left_window_is_bottleneck": left_bottleneck,
                "required_parity_floor_global": required_global,
                "required_parity_floor_local": required_local,
                "overall_over_required_global": overall / required_global,
                "overall_over_required_local": overall / required_local,
                "left_over_required_global": floor_L / required_global,
                "left_over_required_local": floor_L / required_local,
                "right_over_required_global": floor_R / required_global,
                "right_over_required_local": floor_R / required_local,
                "global_threshold_closed": int(overall >= required_global),
                "local_threshold_closed": int(overall >= required_local),
                "remaining_parent_task": (
                    "prove_same_family_piecewise_core_tail_bounds_parent_side"
                ),
            }
        ]
    )

    detail = pd.DataFrame(
        [
            {
                "window": "left",
                "D_range": str(window["best_left_window"]),
                "eta_max": eta_L,
                "lambda_min": lambda_L,
                "core_floor_abs": core_L,
                "induced_floor": floor_L,
                "over_required_global": floor_L / required_global,
                "over_required_local": floor_L / required_local,
            },
            {
                "window": "right",
                "D_range": str(window["best_right_window"]),
                "eta_max": eta_R,
                "lambda_min": lambda_R,
                "core_floor_abs": core_R,
                "induced_floor": floor_R,
                "over_required_global": floor_R / required_global,
                "over_required_local": floor_R / required_local,
            },
        ]
    )

    summary_path = OUTDIR / "chi_open_system_exact_schur_piecewise_parent_lift_source_summary.csv"
    detail_path = OUTDIR / "chi_open_system_exact_schur_piecewise_parent_lift_source_detail.csv"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    print(summary_path)
    print(detail_path)


if __name__ == "__main__":
    main()
