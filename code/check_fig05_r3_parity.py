#!/usr/bin/env python3
"""Check Fig. 05 R3 plotting-path parity against the canonical map table.

Canonical source:
    output/gn_fp_impact/gn_baseline_replacement.csv

This gate protects the submission phase diagram from silent configuration drift:
the live plotting path in code/generate_plots.py must reproduce the current
baseline first-three-layer occupancy fractions before regenerated PNGs are
accepted for the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "output" / "gn_fp_impact" / "gn_baseline_replacement.csv"

# Keep matplotlib/font caches inside the repo when importing generate_plots.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
import generate_plots  # noqa: E402


def load_expected(path: Path) -> dict[str, float]:
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("case") == "baseline_fp_2d_full" and int(row.get("n_max", "0")) == 20:
            return {
                "f_R3_gt_0p90": float(row["f_R3_gt_0p90"]),
                "f_R3_gt_0p95": float(row["f_R3_gt_0p95"]),
                "f_winner_gt_3": float(row["f_winner_gt_3"]),
            }
    raise SystemExit(f"baseline_fp_2d_full / n_max=20 row not found in {path}")


def compute_live() -> dict[str, float]:
    kin = generate_plots.make_baseline_kinetics()
    d_vals = np.linspace(
        generate_plots.PAPER_BASELINE["D_min"],
        generate_plots.PAPER_BASELINE["D_max"],
        generate_plots.PAPER_BASELINE["D_num"],
    )
    eta_vals = np.linspace(0.2, 4.0, 60)
    t_coh = generate_plots.PAPER_BASELINE["t_coh"]
    n_max = generate_plots.PAPER_BASELINE["hll_observable_nmax"]

    r3_values = []
    winner_values = []
    for eta in eta_vals:
        for d in d_vals:
            _, _, meta = kin.get_probabilities(d, eta, t_coh, n_max)
            r3_values.append(float(meta["generation_ratio"]))
            winner_values.append(int(meta["winner"]))

    r3 = np.asarray(r3_values, dtype=float)
    winners = np.asarray(winner_values, dtype=int)
    return {
        "f_R3_gt_0p90": float(np.mean(r3 >= 0.90)),
        "f_R3_gt_0p95": float(np.mean(r3 >= 0.95)),
        "f_winner_gt_3": float(np.mean(winners > 3)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--tol", type=float, default=1e-12)
    args = parser.parse_args()

    expected = load_expected(args.expected_table)
    live = compute_live()
    failures = []
    for key in ("f_R3_gt_0p90", "f_R3_gt_0p95", "f_winner_gt_3"):
        delta = abs(live[key] - expected[key])
        if delta > args.tol:
            failures.append((key, live[key], expected[key], delta))

    if failures:
        for key, live_val, exp_val, delta in failures:
            print(
                f"Fig.05 R3 parity mismatch: {key} live={live_val:.16g}, "
                f"expected={exp_val:.16g}, |delta|={delta:.3e}"
            )
        raise SystemExit(1)

    print("Fig.05 R3 parity check: PASS")
    for key in ("f_R3_gt_0p90", "f_R3_gt_0p95", "f_winner_gt_3"):
        print(f"  {key}: {live[key]:.16g}")


if __name__ == "__main__":
    main()
