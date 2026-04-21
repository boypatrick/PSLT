#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def _window_stats(df: pd.DataFrame) -> dict[str, float]:
    return {
        "D_min": float(df["D"].min()),
        "D_max": float(df["D"].max()),
        "lambda_min": float(df["lambda_total_over_core"].min()),
        "core_floor_abs": float(df["core_abs"].min()),
        "induced_floor": float(df["lambda_total_over_core"].min() * df["core_abs"].min()),
        "exact_floor_abs": float(df["exact_total"].abs().min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--detail-csv",
        default=str(OUTDIR / "chi_open_system_exact_total_core_tightening_source_detail.csv"),
    )
    ap.add_argument(
        "--threshold-summary-csv",
        default=str(OUTDIR / "chi_open_system_exact_schur_lift_parity_threshold_source_summary.csv"),
    )
    args = ap.parse_args()

    detail = pd.read_csv(args.detail_csv).sort_values("D").reset_index(drop=True)
    threshold = pd.read_csv(args.threshold_summary_csv).iloc[0]

    required_global = float(threshold["required_parity_floor_global"])
    required_local = float(threshold["required_parity_floor_local"])

    if not (detail["core_same_sign_as_total"] == 1).all():
        raise RuntimeError("Input detail file is not safe on the core side.")
    if not (detail["tail_opposite_sign_to_total"] == 1).all():
        raise RuntimeError("Input detail file is not safe on the tail side.")

    Ds = [int(round(float(x))) for x in detail["D"].tolist()]
    cuts = range(min(Ds) + 1, max(Ds))

    rows = []
    best = None
    for cut in cuts:
        left = detail[detail["D"] <= cut].copy()
        right = detail[detail["D"] > cut].copy()
        if left.empty or right.empty:
            continue

        left_stats = _window_stats(left)
        right_stats = _window_stats(right)
        overall_floor = min(left_stats["induced_floor"], right_stats["induced_floor"])
        row = {
            "cut_D": float(cut),
            "left_D_min": left_stats["D_min"],
            "left_D_max": left_stats["D_max"],
            "left_lambda_min": left_stats["lambda_min"],
            "left_core_floor_abs": left_stats["core_floor_abs"],
            "left_induced_floor": left_stats["induced_floor"],
            "left_exact_floor_abs": left_stats["exact_floor_abs"],
            "right_D_min": right_stats["D_min"],
            "right_D_max": right_stats["D_max"],
            "right_lambda_min": right_stats["lambda_min"],
            "right_core_floor_abs": right_stats["core_floor_abs"],
            "right_induced_floor": right_stats["induced_floor"],
            "right_exact_floor_abs": right_stats["exact_floor_abs"],
            "overall_induced_floor": overall_floor,
            "overall_over_required_global": overall_floor / required_global,
            "overall_over_required_local": overall_floor / required_local,
        }
        rows.append(row)
        if best is None or row["overall_induced_floor"] > best["overall_induced_floor"]:
            best = row

    if best is None:
        raise RuntimeError("No admissible window split found.")

    scan = pd.DataFrame(rows).sort_values("cut_D").reset_index(drop=True)
    summary = pd.DataFrame(
        [
            {
                "R_star": float(detail["R"].iloc[0]),
                "best_cut_D": float(best["cut_D"]),
                "best_left_window": f"[{int(best['left_D_min'])},{int(best['left_D_max'])}]",
                "best_right_window": f"[{int(best['right_D_min'])},{int(best['right_D_max'])}]",
                "best_left_lambda_min": float(best["left_lambda_min"]),
                "best_left_core_floor_abs": float(best["left_core_floor_abs"]),
                "best_left_induced_floor": float(best["left_induced_floor"]),
                "best_right_lambda_min": float(best["right_lambda_min"]),
                "best_right_core_floor_abs": float(best["right_core_floor_abs"]),
                "best_right_induced_floor": float(best["right_induced_floor"]),
                "best_overall_induced_floor": float(best["overall_induced_floor"]),
                "required_parity_floor_global": required_global,
                "required_parity_floor_local": required_local,
                "best_over_required_global": float(best["overall_over_required_global"]),
                "best_over_required_local": float(best["overall_over_required_local"]),
                "global_threshold_closed": int(best["overall_induced_floor"] >= required_global),
                "local_threshold_closed": int(best["overall_induced_floor"] >= required_local),
            }
        ]
    )

    detail_path = OUTDIR / "chi_open_system_exact_total_core_window_split_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_exact_total_core_window_split_source_summary.csv"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    scan.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
