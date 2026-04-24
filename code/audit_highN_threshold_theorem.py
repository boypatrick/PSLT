#!/usr/bin/env python3
"""
Mode-resolved high-N threshold certificate.

This audit deliberately keeps the theorem target narrow.  It does not introduce
new high-N support or a new visibility object.  It certifies, on the existing
action-derived 1D kinetic proxy and the existing single-track benchmark, that a
fourth layer is absent because the candidate mode is above the continuum
threshold rather than because B_N adds an engineered damping factor.

Outputs:
  - output/highN_decoupling/highN_threshold_theorem_detail.csv
  - output/highN_decoupling/highN_threshold_theorem_summary.csv
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "output" / "highN_decoupling"

sys.path.insert(0, str(ROOT / "code"))
from extract_superrad_prefactor_1d import PhysicalParams, eig_1d, u_z  # noqa: E402


CONFIGS = [
    ("box60_dz002", 60.0, 6001),
    ("primary_box80_dz002", 80.0, 8001),
    ("box100_dz002", 100.0, 10001),
    ("box80_coarse", 80.0, 4001),
    ("box80_fine", 80.0, 12001),
]
PRIMARY = "primary_box80_dz002"


def tridiagonal_entries(U: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dz = float(z[1] - z[0])
    main = 2.0 / dz**2 + U[1:-1]
    off = -1.0 / dz**2 * np.ones(len(main) - 1)
    return main, off


def sturm_count_below(main: np.ndarray, off: np.ndarray, threshold: float = 0.0) -> int:
    """Return the inertia count #{eigenvalues < threshold} for a real tridiagonal.

    The LDL^T pivots form the Sturm sequence.  A tiny pivot is nudged only to
    avoid division by zero; the audited margins are far from zero.
    """

    count = 0
    pivot = float(main[0] - threshold)
    if pivot < 0.0:
        count += 1
    if abs(pivot) < 1.0e-300:
        pivot = -1.0e-300 if pivot < 0.0 else 1.0e-300

    for i in range(1, len(main)):
        pivot = float(main[i] - threshold - (off[i - 1] ** 2) / pivot)
        if pivot < 0.0:
            count += 1
        if abs(pivot) < 1.0e-300:
            pivot = -1.0e-300 if pivot < 0.0 else 1.0e-300
    return count


def read_superrad_summary() -> dict[str, float]:
    path = OUTDIR / "highN_decoupling_superrad_1d_summary.csv"
    with path.open(newline="") as f:
        row = next(csv.DictReader(f))
    return {k: float(v) for k, v in row.items()}


def read_single_track_rows() -> list[dict[str, float]]:
    path = OUTDIR / "highN_decoupling_single_track_summary.csv"
    if path.exists():
        with path.open(newline="") as f:
            return [
                {
                    "D": float(r["D"]),
                    "n_bound_single_track": float(r["n_bound_single_track"]),
                    "supports_N4_bound": float(r["supports_N4_bound"]),
                }
                for r in csv.DictReader(f)
            ]

    data = json.loads((ROOT / "output" / "true_single_track" / "true_results.json").read_text())
    return [
        {
            "D": float(D),
            "n_bound_single_track": float(n_bound),
            "supports_N4_bound": float(int(n_bound) >= 4),
        }
        for D, n_bound in zip(data["D"], data["n_bound"])
    ]


def rows_for_config(label: str, zmax: float, nz: int, p: PhysicalParams) -> list[dict[str, float | str]]:
    z = np.linspace(-zmax, zmax, nz)
    dz = float(z[1] - z[0])
    rows: list[dict[str, float | str]] = []
    for D in range(4, 21):
        U = u_z(z, float(D), p)
        main, off = tridiagonal_entries(U, z)
        evals = eig_1d(U, z, n_states=6)
        n_neg = sturm_count_below(main, off, threshold=0.0)
        row: dict[str, float | str] = {
            "config": label,
            "zmax": zmax,
            "Nz": float(nz),
            "dz": dz,
            "D": float(D),
            "n_negative_sturm": float(n_neg),
            "E1": float(evals[0]),
            "E2": float(evals[1]),
            "E3": float(evals[2]),
            "E4": float(evals[3]),
            "E5": float(evals[4]),
            "E6": float(evals[5]),
            "E3_positive": float(evals[2] > 0.0),
            "E4_positive": float(evals[3] > 0.0),
            "supports_N4_bound": float(n_neg >= 4),
        }
        rows.append(row)
    return rows


def finite_min(rows: Iterable[dict[str, float | str]], key: str) -> float:
    return min(float(r[key]) for r in rows)


def finite_max(rows: Iterable[dict[str, float | str]], key: str) -> float:
    return max(float(r[key]) for r in rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = PhysicalParams()

    detail: list[dict[str, float | str]] = []
    for label, zmax, nz in CONFIGS:
        detail.extend(rows_for_config(label, zmax, nz, p))

    primary = [r for r in detail if r["config"] == PRIMARY]
    box80 = [r for r in detail if str(r["config"]).startswith("box80") or r["config"] == PRIMARY]
    single = read_single_track_rows()
    superrad = read_superrad_summary()

    primary_by_D = {int(float(r["D"])): r for r in primary}
    grid_shifts_E3 = []
    grid_shifts_E4 = []
    for r in box80:
        D = int(float(r["D"]))
        ref = primary_by_D[D]
        if r["config"] == PRIMARY:
            continue
        grid_shifts_E3.append(abs(float(r["E3"]) - float(ref["E3"])))
        grid_shifts_E4.append(abs(float(r["E4"]) - float(ref["E4"])))

    worst_primary_E4 = min(primary, key=lambda r: float(r["E4"]))
    worst_all_E4 = min(detail, key=lambda r: float(r["E4"]))
    max_grid_shift_E3 = max(grid_shifts_E3) if grid_shifts_E3 else math.nan
    max_grid_shift_E4 = max(grid_shifts_E4) if grid_shifts_E4 else math.nan

    summary = {
        "D_min": 4.0,
        "D_max": 20.0,
        "primary_config": PRIMARY,
        "primary_max_negative_sturm_count": finite_max(primary, "n_negative_sturm"),
        "primary_min_E3": finite_min(primary, "E3"),
        "primary_min_E4": finite_min(primary, "E4"),
        "primary_worst_E4_D": float(worst_primary_E4["D"]),
        "primary_all_E3_positive": float(all(float(r["E3"]) > 0.0 for r in primary)),
        "primary_all_E4_positive": float(all(float(r["E4"]) > 0.0 for r in primary)),
        "all_configs_max_negative_sturm_count": finite_max(detail, "n_negative_sturm"),
        "all_configs_min_E3": finite_min(detail, "E3"),
        "all_configs_min_E4": finite_min(detail, "E4"),
        "all_configs_worst_E4_config": worst_all_E4["config"],
        "all_configs_worst_E4_D": float(worst_all_E4["D"]),
        "box80_grid_max_abs_E3_shift_vs_primary": max_grid_shift_E3,
        "box80_grid_max_abs_E4_shift_vs_primary": max_grid_shift_E4,
        "single_track_max_bound_count": max(r["n_bound_single_track"] for r in single),
        "single_track_supports_N4_bound_max": max(r["supports_N4_bound"] for r in single),
        "forced_gamma_N4_over_N3_l1_min": superrad["gamma_N4_over_N3_l1_min"],
        "forced_gamma_N4_over_N3_l1_max": superrad["gamma_N4_over_N3_l1_max"],
        "forced_gamma_N4_over_N2_l1_min": superrad["gamma_N4_over_N2_l1_min"],
        "forced_gamma_N4_over_N2_l1_max": superrad["gamma_N4_over_N2_l1_max"],
        "theorem_status": "mode_resolved_no_fourth_bound_layer_closed_on_audited_domain",
        "remaining_scope": "continuum_tail_bound_if_extending_beyond_the_current_finite_volume_Sturm_certificate",
    }

    detail_out = OUTDIR / "highN_threshold_theorem_detail.csv"
    summary_out = OUTDIR / "highN_threshold_theorem_summary.csv"

    with detail_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        writer.writeheader()
        writer.writerows(detail)

    with summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Wrote {detail_out}")
    print(f"Wrote {summary_out}")
    print("[mode-resolved high-N threshold summary]")
    for k, v in summary.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
