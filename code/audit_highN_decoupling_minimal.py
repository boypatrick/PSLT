#!/usr/bin/env python3
"""
Minimal audit for the physical origin of higher-generation decoupling.

This script keeps the question deliberately narrow:

1. In the existing action-derived single-track benchmark, does the bound sector
   ever support a fourth bound level?
2. In the 1D kinetic/superradiant candidate chain, is N=4 better interpreted as
   a bound level with exponentially tiny formation, or as a state already above
   continuum threshold?
3. If one nevertheless forces the N=4 channel into the geometric-rate formula,
   is the dominant effect "huge barrier suppression" or "loss of binding"?

Outputs:
  - highN_decoupling_single_track_summary.csv
  - highN_decoupling_superrad_1d_detail.csv
  - highN_decoupling_superrad_1d_summary.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "highN_decoupling"


def load_single_track_summary() -> List[Dict[str, float]]:
    path = ROOT / "output" / "true_single_track" / "true_results.json"
    data = json.loads(path.read_text())
    rows: List[Dict[str, float]] = []
    for D, n_bound, E1, omega1 in zip(data["D"], data["n_bound"], data["E_bound"], data["omega"]):
        rows.append(
            {
                "D": float(D),
                "n_bound_single_track": int(n_bound),
                "E1_single_track": float(E1),
                "omega1_single_track": float(omega1),
                "supports_N4_bound": int(int(n_bound) >= 4),
            }
        )
    return rows


def compute_superrad_rows() -> List[Dict[str, float]]:
    import sys

    sys.path.insert(0, str(ROOT / "code"))
    from extract_superrad_prefactor_1d import PhysicalParams, Level, eig_1d, solve_case, u_z

    p = PhysicalParams()
    zmax = 80.0
    level = Level("fine", 8001)
    z = np.linspace(-zmax, zmax, level.Nz)

    rows: List[Dict[str, float]] = []
    for D in range(4, 21):
        U = u_z(z, float(D), p)
        Evals = eig_1d(U, z, n_states=6)
        n_bound = int((Evals < 0.0).sum())

        row: Dict[str, float] = {
            "D": float(D),
            "n_bound_superrad_1d": n_bound,
            "E1": float(Evals[0]),
            "E2": float(Evals[1]),
            "E3": float(Evals[2]),
            "E4": float(Evals[3]),
            "E5": float(Evals[4]),
            "E6": float(Evals[5]),
            "E4_positive": int(Evals[3] > 0.0),
            "E3_positive": int(Evals[2] > 0.0),
        }

        for N in (2, 3, 4):
            case = solve_case(D=float(D), level=level, zmax=zmax, p=p, n_ref=N)
            row[f"omega_N{N}"] = float(case["omega_ref"])
            row[f"S_l1_N{N}"] = float(case["S_l1"])
            row[f"gamma_geo_l1_N{N}"] = float(case["gamma_geo_l1"])

        row["gamma_N4_over_N3_l1"] = row["gamma_geo_l1_N4"] / max(row["gamma_geo_l1_N3"], 1e-300)
        row["gamma_N4_over_N2_l1"] = row["gamma_geo_l1_N4"] / max(row["gamma_geo_l1_N2"], 1e-300)
        rows.append(row)
    return rows


def summarize_superrad(rows: List[Dict[str, float]]) -> Dict[str, float]:
    E4 = np.array([row["E4"] for row in rows], dtype=float)
    E3 = np.array([row["E3"] for row in rows], dtype=float)
    n_bound = np.array([row["n_bound_superrad_1d"] for row in rows], dtype=int)
    gamma43 = np.array([row["gamma_N4_over_N3_l1"] for row in rows], dtype=float)
    gamma42 = np.array([row["gamma_N4_over_N2_l1"] for row in rows], dtype=float)

    return {
        "D_min": float(rows[0]["D"]),
        "D_max": float(rows[-1]["D"]),
        "max_n_bound_superrad_1d": int(np.max(n_bound)),
        "min_E3": float(np.min(E3)),
        "max_E3": float(np.max(E3)),
        "min_E4": float(np.min(E4)),
        "max_E4": float(np.max(E4)),
        "all_E4_positive": int(np.all(E4 > 0.0)),
        "all_E3_positive": int(np.all(E3 > 0.0)),
        "gamma_N4_over_N3_l1_min": float(np.min(gamma43)),
        "gamma_N4_over_N3_l1_max": float(np.max(gamma43)),
        "gamma_N4_over_N2_l1_min": float(np.min(gamma42)),
        "gamma_N4_over_N2_l1_max": float(np.max(gamma42)),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    single_track_rows = load_single_track_summary()
    superrad_rows = compute_superrad_rows()
    superrad_summary = summarize_superrad(superrad_rows)

    out_single = OUTDIR / "highN_decoupling_single_track_summary.csv"
    out_superrad = OUTDIR / "highN_decoupling_superrad_1d_detail.csv"
    out_superrad_summary = OUTDIR / "highN_decoupling_superrad_1d_summary.csv"

    with out_single.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(single_track_rows[0].keys()))
        writer.writeheader()
        writer.writerows(single_track_rows)

    with out_superrad.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(superrad_rows[0].keys()))
        writer.writeheader()
        writer.writerows(superrad_rows)

    with out_superrad_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(superrad_summary.keys()))
        writer.writeheader()
        writer.writerow(superrad_summary)

    print(f"[saved] {out_single}")
    print(f"[saved] {out_superrad}")
    print(f"[saved] {out_superrad_summary}")
    print("\n[single-track bound-count summary]")
    for row in single_track_rows:
        print(
            f"D={int(row['D']):2d}  n_bound={row['n_bound_single_track']}  "
            f"supports_N4_bound={row['supports_N4_bound']}"
        )
    print("\n[superrad 1D summary]")
    for key, value in superrad_summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
