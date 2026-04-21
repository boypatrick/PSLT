#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"


def _stats(df: pd.DataFrame) -> dict[str, float]:
    return {
        "D_min": float(df["D"].min()),
        "D_max": float(df["D"].max()),
        "eta_max": float(df["eta_total_tail_over_core"].max()),
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

    left = detail[detail["D"] <= 11].copy()
    if left.empty:
        raise RuntimeError("No left-window rows found.")

    required_global = float(threshold["required_parity_floor_global"])
    required_local = float(threshold["required_parity_floor_local"])

    cuts = range(int(left["D"].min()) + 1, int(left["D"].max()))
    rows = []
    best = None
    for cut in cuts:
        a = left[left["D"] <= cut].copy()
        b = left[left["D"] > cut].copy()
        if a.empty or b.empty:
            continue
        sa = _stats(a)
        sb = _stats(b)
        overall = min(sa["induced_floor"], sb["induced_floor"])
        row = {
            "cut_D": float(cut),
            "leftA_window": f"[{int(sa['D_min'])},{int(sa['D_max'])}]",
            "leftA_eta_max": sa["eta_max"],
            "leftA_lambda_min": sa["lambda_min"],
            "leftA_core_floor_abs": sa["core_floor_abs"],
            "leftA_induced_floor": sa["induced_floor"],
            "leftA_exact_floor_abs": sa["exact_floor_abs"],
            "leftB_window": f"[{int(sb['D_min'])},{int(sb['D_max'])}]",
            "leftB_eta_max": sb["eta_max"],
            "leftB_lambda_min": sb["lambda_min"],
            "leftB_core_floor_abs": sb["core_floor_abs"],
            "leftB_induced_floor": sb["induced_floor"],
            "leftB_exact_floor_abs": sb["exact_floor_abs"],
            "overall_induced_floor": overall,
            "overall_over_required_global": overall / required_global,
            "overall_over_required_local": overall / required_local,
        }
        rows.append(row)
        if best is None or overall > best["overall_induced_floor"]:
            best = row

    if best is None:
        raise RuntimeError("No admissible left-window split found.")

    eta_max_row = left.loc[left["eta_total_tail_over_core"].idxmax()]
    core_floor_row = left.loc[left["core_abs"].idxmin()]

    summary = pd.DataFrame(
        [
            {
                "best_cut_D": float(best["cut_D"]),
                "best_leftA_window": best["leftA_window"],
                "best_leftA_eta_max": float(best["leftA_eta_max"]),
                "best_leftA_core_floor_abs": float(best["leftA_core_floor_abs"]),
                "best_leftA_induced_floor": float(best["leftA_induced_floor"]),
                "best_leftB_window": best["leftB_window"],
                "best_leftB_eta_max": float(best["leftB_eta_max"]),
                "best_leftB_core_floor_abs": float(best["leftB_core_floor_abs"]),
                "best_leftB_induced_floor": float(best["leftB_induced_floor"]),
                "best_overall_induced_floor": float(best["overall_induced_floor"]),
                "required_parity_floor_global": required_global,
                "required_parity_floor_local": required_local,
                "best_over_required_global": float(best["overall_over_required_global"]),
                "best_over_required_local": float(best["overall_over_required_local"]),
                "eta_left_max_D": float(eta_max_row["D"]),
                "eta_left_max_value": float(eta_max_row["eta_total_tail_over_core"]),
                "core_left_floor_D": float(core_floor_row["D"]),
                "core_left_floor_value": float(core_floor_row["core_abs"]),
                "split_decouples_eta_and_core_bottlenecks": int(
                    int(round(float(eta_max_row["D"]))) != int(round(float(core_floor_row["D"])))
                ),
            }
        ]
    )

    detail_path = OUTDIR / "chi_open_system_exact_schur_left_window_split_source_detail.csv"
    summary_path = OUTDIR / "chi_open_system_exact_schur_left_window_split_source_summary.csv"
    pd.DataFrame(rows).to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
