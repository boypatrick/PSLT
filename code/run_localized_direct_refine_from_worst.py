#!/usr/bin/env python3
"""
Build a local-refinement D-grid from worst drift points and run
scan_localized_direct_surface_bias.py on that explicit D grid.

This is a deterministic refinement utility (no new ML dependencies).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SCAN_SCRIPT = ROOT / "code" / "scan_localized_direct_surface_bias.py"
DEFAULT_WORST = ROOT / "output" / "kinetic_action_chain" / "full_direct_worst_points_table.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run localized-direct local refinement from worst-points table.")
    ap.add_argument("--worst-csv", type=str, default=str(DEFAULT_WORST))
    ap.add_argument("--scenario-contains", type=str, default="release_tuned")
    ap.add_argument("--top-k-d", type=int, default=10, help="Number of unique high-drift D anchors used.")
    ap.add_argument("--neighbor-step", type=float, default=0.25, help="Neighbor spacing added around each selected D.")
    ap.add_argument("--neighbor-levels", type=int, default=1, help="Number of +/- neighbor shells.")
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=20.0)
    ap.add_argument("--eta-min", type=float, default=0.2)
    ap.add_argument("--eta-max", type=float, default=4.0)
    ap.add_argument("--eta-num", type=int, default=41)
    ap.add_argument("--tag-suffix", type=str, default="refine_from_worst")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    worst_csv = Path(args.worst_csv)
    if not worst_csv.exists():
        raise FileNotFoundError(f"Worst-points CSV not found: {worst_csv}")

    df = pd.read_csv(worst_csv)
    if "scenario" not in df.columns or "D" not in df.columns or "abs_delta_mu_mumu" not in df.columns:
        raise RuntimeError("Worst-points CSV missing required columns: scenario, D, abs_delta_mu_mumu.")

    sub = df[df["scenario"].astype(str).str.contains(str(args.scenario_contains), case=False, regex=False)].copy()
    if len(sub) == 0:
        raise RuntimeError(f"No rows matched scenario filter: {args.scenario_contains}")

    sub = sub.sort_values("abs_delta_mu_mumu", ascending=False).reset_index(drop=True)
    d_selected: List[float] = []
    seen = set()
    for d in sub["D"].astype(float).to_list():
        k = round(float(d), 8)
        if k in seen:
            continue
        seen.add(k)
        d_selected.append(float(d))
        if len(d_selected) >= int(args.top_k_d):
            break
    if len(d_selected) < 2:
        raise RuntimeError("Insufficient unique D anchors from worst-points table.")

    d_aug = set([float(args.d_min), float(args.d_max)])
    step = float(args.neighbor_step)
    levels = int(max(0, args.neighbor_levels))
    for d in d_selected:
        d_aug.add(float(d))
        for lv in range(1, levels + 1):
            d_aug.add(float(np.clip(d + lv * step, float(args.d_min), float(args.d_max))))
            d_aug.add(float(np.clip(d - lv * step, float(args.d_min), float(args.d_max))))

    d_vals = sorted(d_aug)
    if len(d_vals) < 2:
        raise RuntimeError("Refined D grid collapsed to <2 points.")

    d_values_arg = ",".join(f"{x:.10g}" for x in d_vals)
    cmd = [
        sys.executable,
        str(SCAN_SCRIPT),
        "--d-values",
        d_values_arg,
        "--eta-min",
        str(float(args.eta_min)),
        "--eta-max",
        str(float(args.eta_max)),
        "--eta-num",
        str(int(args.eta_num)),
        "--tag-suffix",
        str(args.tag_suffix),
    ]
    print(f"[refine] selected D count={len(d_vals)}")
    print(f"[refine] D values={d_values_arg}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()

