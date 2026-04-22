#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "chi_open_system"


def read_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def main() -> None:
    detail_path = OUT_DIR / "chi_open_system_exact_total_core_tightening_source_detail.csv"
    threshold_path = OUT_DIR / "chi_open_system_exact_schur_lift_parity_threshold_source_summary.csv"
    hier_path = OUT_DIR / "chi_open_system_exact_schur_left_window_hierarchical_split_source_summary.csv"

    threshold = read_csv_row(threshold_path)
    hier = read_csv_row(hier_path)

    with detail_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    d11 = next(row for row in rows if int(round(float(row["D"]))) == 11)

    parity_required_global = float(threshold["required_parity_floor_global"])
    parity_required_local = float(threshold["required_parity_floor_local"])

    exact_total = abs(float(d11["exact_total"]))
    core_abs = float(d11["core_abs"])
    tail_abs = float(d11["tail_abs"])
    eta_11 = float(d11["eta_total_tail_over_core"])
    lambda_11 = float(d11["lambda_total_over_core"])

    required_core_global = parity_required_global / lambda_11
    required_core_local = parity_required_local / lambda_11

    summary = {
        "D_bottleneck": 11.0,
        "R_star": float(d11["R"]),
        "eta_11": eta_11,
        "lambda_11": lambda_11,
        "core_abs_11": core_abs,
        "tail_abs_11": tail_abs,
        "exact_total_abs_11": exact_total,
        "required_parity_floor_global": parity_required_global,
        "required_parity_floor_local": parity_required_local,
        "required_core_floor_global": required_core_global,
        "required_core_floor_local": required_core_local,
        "core_over_required_global": core_abs / required_core_global,
        "core_over_required_local": core_abs / required_core_local,
        "tail_over_exact_total_11": tail_abs / exact_total,
        "core_over_exact_total_11": core_abs / exact_total,
        "interval_floor_4_6": float(hier["best_A_induced_floor"]),
        "interval_floor_7_10": float(hier["best_B_induced_floor"]),
        "singleton11_is_overall_bottleneck": float(hier["singleton11_is_overall_bottleneck"]),
    }

    detail_rows = [
        {
            "quantity": "exact_total_abs_11",
            "value": exact_total,
            "role": "singleton_floor",
        },
        {
            "quantity": "core_abs_11",
            "value": core_abs,
            "role": "audited_core",
        },
        {
            "quantity": "tail_abs_11",
            "value": tail_abs,
            "role": "opposite_sign_tail",
        },
        {
            "quantity": "required_core_floor_global",
            "value": required_core_global,
            "role": "global_threshold_target",
        },
        {
            "quantity": "required_core_floor_local",
            "value": required_core_local,
            "role": "local_threshold_target",
        },
        {
            "quantity": "interval_floor_4_6",
            "value": float(hier["best_A_induced_floor"]),
            "role": "non_bottleneck_interval",
        },
        {
            "quantity": "interval_floor_7_10",
            "value": float(hier["best_B_induced_floor"]),
            "role": "non_bottleneck_interval",
        },
    ]

    summary_path = OUT_DIR / "chi_open_system_exact_schur_singleton_knot_source_summary.csv"
    detail_out_path = OUT_DIR / "chi_open_system_exact_schur_singleton_knot_source_detail.csv"

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    with detail_out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)

    print(summary_path)
    print(detail_out_path)


if __name__ == "__main__":
    main()
