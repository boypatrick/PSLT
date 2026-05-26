#!/usr/bin/env python3
"""Spectral projector transport/coherence gate for N=1,2,3 along D_sep.

This script is intentionally narrow.  It consumes the existing G1 low-mode
spectral artifact and checks whether the audited finite-grid eigenvalue branches
are separated enough to support a conditional Kato/Riesz projector-transport
theorem.

It does not recompute the operator, does not modify the release baseline, and
does not claim a continuum theorem by itself.  The rigorous theorem is
conditional on a C^1 norm-resolvent family with an open gap.  This gate records
the finite-grid evidence: strict low-mode gaps, stable rank counts, Riesz contour
radius floors, and adjacent-D displacement/gap ratios.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPECTRUM = ROOT / "output" / "gn_fp_2d" / "gn_phase_space_2d_spectrum_Dgrid21.csv"
DEFAULT_COUNT = ROOT / "output" / "gn_fp_2d" / "gn_lowN_microstate_count_detail.csv"
OUTDIR = ROOT / "output" / "spectral_transport"
SUMMARY_CSV = OUTDIR / "spectral_projector_transport_summary.csv"
DETAIL_CSV = OUTDIR / "spectral_projector_transport_detail.csv"
STEP_CSV = OUTDIR / "spectral_projector_transport_steps.csv"
DETAIL_JSON = OUTDIR / "spectral_projector_transport_detail.json"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def load_low_modes(spectrum_csv: Path, level: str, modes_needed: int) -> dict[float, np.ndarray]:
    spec = pd.read_csv(spectrum_csv)
    spec = spec[spec["level"].astype(str) == str(level)].copy()
    if spec.empty:
        raise RuntimeError(f"no spectrum rows for level={level!r} in {spectrum_csv}")

    out: dict[float, np.ndarray] = {}
    for D, sub in spec.groupby("D"):
        sub = sub.sort_values("mode_n")
        vals = sub["lambda_n"].to_numpy(dtype=float)
        if len(vals) < modes_needed:
            raise RuntimeError(f"D={D}: need at least {modes_needed} eigenvalues, got {len(vals)}")
        out[float(D)] = vals[:modes_needed]
    return dict(sorted(out.items()))


def nearest_gap(lam: np.ndarray, mode_index0: int) -> float:
    left = lam[mode_index0] - lam[mode_index0 - 1] if mode_index0 > 0 else np.inf
    right = lam[mode_index0 + 1] - lam[mode_index0] if mode_index0 + 1 < len(lam) else np.inf
    return float(min(left, right))


def finite_slopes(D_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.diff(values) / np.diff(D_values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument("--count-csv", type=Path, default=DEFAULT_COUNT)
    parser.add_argument("--level", default="fine")
    parser.add_argument("--modes-needed", type=int, default=4, help="Need mode 4 to isolate the N=3 projector.")
    parser.add_argument("--contour-fraction", type=float, default=0.45)
    parser.add_argument(
        "--max-step-gap-ratio-tol",
        type=float,
        default=0.25,
        help="Observed adjacent-grid displacement divided by local nearest gap must stay below this value.",
    )
    parser.add_argument("--summary", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--detail", type=Path, default=DETAIL_CSV)
    parser.add_argument("--steps", type=Path, default=STEP_CSV)
    parser.add_argument("--detail-json", type=Path, default=DETAIL_JSON)
    args = parser.parse_args()

    spectrum_csv = args.spectrum_csv if args.spectrum_csv.is_absolute() else ROOT / args.spectrum_csv
    count_csv = args.count_csv if args.count_csv.is_absolute() else ROOT / args.count_csv
    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary
    detail_path = args.detail if args.detail.is_absolute() else ROOT / args.detail
    step_path = args.steps if args.steps.is_absolute() else ROOT / args.steps
    json_path = args.detail_json if args.detail_json.is_absolute() else ROOT / args.detail_json

    branches = load_low_modes(spectrum_csv, args.level, args.modes_needed)
    D_values = np.asarray(sorted(branches), dtype=float)
    lambdas = np.asarray([branches[float(D)] for D in D_values], dtype=float)

    count = pd.read_csv(count_csv)
    count = count[count["level"].astype(str) == str(args.level)].copy()
    if count.empty:
        raise RuntimeError(f"no count rows for level={args.level!r} in {count_csv}")
    rank_pass = bool(count["row_pass"].all())

    detail_rows: list[dict[str, Any]] = []
    for iD, D in enumerate(D_values):
        lam = lambdas[iD]
        gap12 = float(lam[1] - lam[0])
        gap23 = float(lam[2] - lam[1])
        gap34 = float(lam[3] - lam[2])
        for n in (1, 2, 3):
            gap = nearest_gap(lam, n - 1)
            detail_rows.append(
                {
                    "D": float(D),
                    "N": n,
                    "lambda_N": float(lam[n - 1]),
                    "gap12": gap12,
                    "gap23": gap23,
                    "gap34": gap34,
                    "nearest_gap_N": gap,
                    "riesz_radius_N": float(args.contour_fraction * gap),
                    "contour_fraction": float(args.contour_fraction),
                    "strict_gap_pass": bool(gap > 0.0),
                }
            )

    step_rows: list[dict[str, Any]] = []
    for i in range(len(D_values) - 1):
        D0 = float(D_values[i])
        D1 = float(D_values[i + 1])
        lam0 = lambdas[i]
        lam1 = lambdas[i + 1]
        for n in (1, 2, 3):
            gap0 = nearest_gap(lam0, n - 1)
            gap1 = nearest_gap(lam1, n - 1)
            local_gap = min(gap0, gap1)
            displacement = abs(float(lam1[n - 1] - lam0[n - 1]))
            ratio = displacement / max(local_gap, 1.0e-300)
            step_rows.append(
                {
                    "D_left": D0,
                    "D_right": D1,
                    "N": n,
                    "lambda_left": float(lam0[n - 1]),
                    "lambda_right": float(lam1[n - 1]),
                    "delta_D": float(D1 - D0),
                    "lambda_displacement": displacement,
                    "local_nearest_gap_floor": float(local_gap),
                    "step_gap_ratio": float(ratio),
                    "step_ratio_pass": bool(ratio <= args.max_step_gap_ratio_tol),
                }
            )

    min_gap_by_mode = {
        n: min(float(r["nearest_gap_N"]) for r in detail_rows if int(r["N"]) == n) for n in (1, 2, 3)
    }
    min_radius_by_mode = {
        n: min(float(r["riesz_radius_N"]) for r in detail_rows if int(r["N"]) == n) for n in (1, 2, 3)
    }
    max_step_ratio_by_mode = {
        n: max(float(r["step_gap_ratio"]) for r in step_rows if int(r["N"]) == n) for n in (1, 2, 3)
    }
    slope_by_mode = {}
    total_variation_by_mode = {}
    for n in (1, 2, 3):
        vals = lambdas[:, n - 1]
        slopes = finite_slopes(D_values, vals)
        slope_by_mode[n] = float(np.max(np.abs(slopes)))
        total_variation_by_mode[n] = float(np.sum(np.abs(np.diff(vals))))

    all_strict_gap = all(bool(r["strict_gap_pass"]) for r in detail_rows)
    all_step_ratio = all(bool(r["step_ratio_pass"]) for r in step_rows)
    verdict = (
        "CONDITIONAL_PROJECTOR_TRANSPORT_COHERENCE_SUPPORTED_ON_G1_GRID"
        if all_strict_gap and rank_pass and all_step_ratio
        else "PROJECTOR_TRANSPORT_COHERENCE_DIAGNOSTIC_ONLY"
    )

    summary_row = {
        "gate": "T1",
        "verdict": verdict,
        "level": args.level,
        "n_D": int(len(D_values)),
        "D_min": float(np.min(D_values)),
        "D_max": float(np.max(D_values)),
        "delta_D_min": float(np.min(np.diff(D_values))),
        "delta_D_max": float(np.max(np.diff(D_values))),
        "contour_fraction": float(args.contour_fraction),
        "max_step_gap_ratio_tol": float(args.max_step_gap_ratio_tol),
        "rank_count_rows_pass": rank_pass,
        "all_strict_gap_pass": all_strict_gap,
        "all_step_ratio_pass": all_step_ratio,
        "min_nearest_gap_N1": min_gap_by_mode[1],
        "min_nearest_gap_N2": min_gap_by_mode[2],
        "min_nearest_gap_N3": min_gap_by_mode[3],
        "min_riesz_radius_N1": min_radius_by_mode[1],
        "min_riesz_radius_N2": min_radius_by_mode[2],
        "min_riesz_radius_N3": min_radius_by_mode[3],
        "max_step_gap_ratio_N1": max_step_ratio_by_mode[1],
        "max_step_gap_ratio_N2": max_step_ratio_by_mode[2],
        "max_step_gap_ratio_N3": max_step_ratio_by_mode[3],
        "max_abs_lambda_slope_N1": slope_by_mode[1],
        "max_abs_lambda_slope_N2": slope_by_mode[2],
        "max_abs_lambda_slope_N3": slope_by_mode[3],
        "total_variation_lambda_N1": total_variation_by_mode[1],
        "total_variation_lambda_N2": total_variation_by_mode[2],
        "total_variation_lambda_N3": total_variation_by_mode[3],
    }

    write_csv(summary_path, [summary_row])
    write_csv(detail_path, detail_rows)
    write_csv(step_path, step_rows)

    detail_json = {
        "claim_firewall": {
            "baseline_change": False,
            "main_tex_change": False,
            "continuum_theorem_claimed_by_script": False,
            "interpretation": (
                "finite-grid support for a conditional Kato/Riesz projector transport theorem; "
                "the continuum theorem still requires an analytic open-gap assumption between audited D knots"
            ),
        },
        "theorem_inputs_checked": {
            "strict_grid_gaps": all_strict_gap,
            "rank_count_rows_pass": rank_pass,
            "adjacent_displacement_over_gap_pass": all_step_ratio,
        },
        "summary": summary_row,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(detail_json, f, indent=2)
        f.write("\n")

    print(verdict)
    print(f"summary={summary_path}")
    print(f"detail={detail_path}")
    print(f"steps={step_path}")
    print(f"detail_json={json_path}")
    print(
        "min_gap_N=(%.12g, %.12g, %.12g) max_step_ratio_N=(%.12g, %.12g, %.12g)"
        % (
            min_gap_by_mode[1],
            min_gap_by_mode[2],
            min_gap_by_mode[3],
            max_step_ratio_by_mode[1],
            max_step_ratio_by_mode[2],
            max_step_ratio_by_mode[3],
        )
    )


if __name__ == "__main__":
    main()
