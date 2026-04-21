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
        "--piecewise-summary-csv",
        default=str(OUTDIR / "chi_open_system_exact_schur_piecewise_parent_lift_source_summary.csv"),
    )
    args = ap.parse_args()

    s = pd.read_csv(args.piecewise_summary_csv).iloc[0]

    eta_left = float(s["eta_left_max"])
    lambda_left = float(s["lambda_left_min"])
    core_left = float(s["core_left_floor_abs"])
    induced_left = float(s["induced_left_floor"])

    eta_right = float(s["eta_right_max"])
    core_right = float(s["core_right_floor_abs"])
    induced_right = float(s["induced_right_floor"])

    required_global = float(s["required_parity_floor_global"])
    required_local = float(s["required_parity_floor_local"])

    rows = []
    summary = {
        "eta_left_actual": eta_left,
        "lambda_left_actual": lambda_left,
        "core_left_actual": core_left,
        "induced_left_actual": induced_left,
        "eta_right_actual": eta_right,
        "core_right_actual": core_right,
        "induced_right_actual": induced_right,
        "required_parity_floor_global": required_global,
        "required_parity_floor_local": required_local,
        "left_window_only_bottleneck": int(induced_left <= induced_right),
        "right_window_nonbottleneck_global": int(induced_right >= required_global),
        "right_window_nonbottleneck_local": int(induced_right >= required_local),
    }

    for label, required in (("global", required_global), ("local", required_local)):
        eta_req = 1.0 - required / core_left
        core_req = required / (1.0 - eta_left)
        summary[f"eta_left_required_{label}"] = eta_req
        summary[f"eta_left_slack_{label}"] = eta_req - eta_left
        summary[f"eta_left_required_over_actual_{label}"] = eta_req / eta_left
        summary[f"core_left_required_{label}"] = core_req
        summary[f"core_left_actual_over_required_{label}"] = core_left / core_req
        summary[f"left_actual_over_required_{label}"] = induced_left / required
        rows.append(
            {
                "threshold_choice": label,
                "required_parity_floor": required,
                "eta_left_actual": eta_left,
                "eta_left_required_if_core_fixed": eta_req,
                "eta_left_slack": eta_req - eta_left,
                "eta_left_required_over_actual": eta_req / eta_left,
                "core_left_actual": core_left,
                "core_left_required_if_eta_fixed": core_req,
                "core_left_actual_over_required": core_left / core_req,
                "induced_left_actual": induced_left,
                "induced_left_over_required": induced_left / required,
            }
        )

    summary_path = OUTDIR / "chi_open_system_exact_schur_left_window_bottleneck_source_summary.csv"
    detail_path = OUTDIR / "chi_open_system_exact_schur_left_window_bottleneck_source_detail.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame(rows).to_csv(detail_path, index=False)
    print(summary_path)
    print(detail_path)


if __name__ == "__main__":
    main()
