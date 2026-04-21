#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def _stats(df: pd.DataFrame) -> dict[str, float]:
    eta_max = float(df["eta_total_tail_over_core"].max())
    lambda_min = float(df["lambda_total_over_core"].min())
    core_floor_abs = float(df["core_abs"].min())
    induced_floor = float(lambda_min * core_floor_abs)
    exact_floor_abs = float(df["exact_total"].abs().min())
    return {
        "D_min": float(df["D"].min()),
        "D_max": float(df["D"].max()),
        "eta_max": eta_max,
        "lambda_min": lambda_min,
        "core_floor_abs": core_floor_abs,
        "induced_floor": induced_floor,
        "exact_floor_abs": exact_floor_abs,
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

    left1 = detail[(detail["D"] >= 4) & (detail["D"] <= 10)].copy()
    if left1.empty:
        raise RuntimeError("No rows found on the first left subwindow [4,10].")

    required_global = float(threshold["required_parity_floor_global"])
    required_local = float(threshold["required_parity_floor_local"])

    cuts = range(int(left1["D"].min()), int(left1["D"].max()))
    rows = []
    best = None
    for cut in cuts:
        a = left1[left1["D"] <= cut].copy()
        b = left1[left1["D"] > cut].copy()
        if a.empty or b.empty:
            continue
        sa = _stats(a)
        sb = _stats(b)
        overall = min(sa["induced_floor"], sb["induced_floor"])
        row = {
            "cut_D": float(cut),
            "window_A": f"[{int(sa['D_min'])},{int(sa['D_max'])}]",
            "A_eta_max": sa["eta_max"],
            "A_lambda_min": sa["lambda_min"],
            "A_core_floor_abs": sa["core_floor_abs"],
            "A_induced_floor": sa["induced_floor"],
            "A_exact_floor_abs": sa["exact_floor_abs"],
            "window_B": f"[{int(sb['D_min'])},{int(sb['D_max'])}]",
            "B_eta_max": sb["eta_max"],
            "B_lambda_min": sb["lambda_min"],
            "B_core_floor_abs": sb["core_floor_abs"],
            "B_induced_floor": sb["induced_floor"],
            "B_exact_floor_abs": sb["exact_floor_abs"],
            "overall_induced_floor": overall,
            "overall_over_required_global": overall / required_global,
            "overall_over_required_local": overall / required_local,
        }
        rows.append(row)
        if best is None or overall > best["overall_induced_floor"]:
            best = row

    if best is None:
        raise RuntimeError("No admissible hierarchical split found on [4,10].")

    singleton11 = detail[detail["D"] == 11].copy()
    if singleton11.empty:
        raise RuntimeError("No D=11 row found.")
    s11 = _stats(singleton11)

    overall_hierarchical_floor = min(best["overall_induced_floor"], s11["induced_floor"])

    summary = pd.DataFrame(
        [
            {
                "best_secondary_cut_D": float(best["cut_D"]),
                "best_window_A": best["window_A"],
                "best_A_eta_max": float(best["A_eta_max"]),
                "best_A_core_floor_abs": float(best["A_core_floor_abs"]),
                "best_A_induced_floor": float(best["A_induced_floor"]),
                "best_window_B": best["window_B"],
                "best_B_eta_max": float(best["B_eta_max"]),
                "best_B_core_floor_abs": float(best["B_core_floor_abs"]),
                "best_B_induced_floor": float(best["B_induced_floor"]),
                "singleton11_eta": s11["eta_max"],
                "singleton11_core_floor_abs": s11["core_floor_abs"],
                "singleton11_induced_floor": s11["induced_floor"],
                "overall_hierarchical_floor": overall_hierarchical_floor,
                "required_parity_floor_global": required_global,
                "required_parity_floor_local": required_local,
                "overall_over_required_global": overall_hierarchical_floor / required_global,
                "overall_over_required_local": overall_hierarchical_floor / required_local,
                "singleton11_is_overall_bottleneck": int(
                    abs(overall_hierarchical_floor - s11["induced_floor"]) <= 1e-18
                ),
            }
        ]
    )

    detail_path = OUTDIR / "chi_open_system_exact_schur_left_window_hierarchical_split_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_exact_schur_left_window_hierarchical_split_source_summary.csv"
    pd.DataFrame(rows).to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
