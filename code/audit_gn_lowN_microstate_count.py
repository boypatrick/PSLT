#!/usr/bin/env python3
"""
Finite-volume low-N microstate-count audit for the g_N phase-space chain.

This script does not change the release baseline.  It certifies the exact
finite-dimensional spectral count behind the low modes and records how that
integer projector count differs from the bounded shell-volume weight currently
used by the fp_2d_full baseline.

Inputs:
  - output/gn_fp_2d/gn_phase_space_2d_Dgrid21.csv
  - output/gn_fp_2d/gn_phase_space_2d_spectrum_Dgrid21.csv

Outputs:
  - output/gn_fp_2d/gn_lowN_microstate_count_detail.csv
  - output/gn_fp_2d/gn_lowN_microstate_count_summary.csv
  - paper/gn_lowN_microstate_count_detail.csv
  - paper/gn_lowN_microstate_count_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "output" / "gn_fp_2d"
PAPER_DIR = ROOT / "paper"


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_count(lambdas: np.ndarray, threshold: float, tol: float) -> int:
    return int(np.count_nonzero(lambdas <= threshold + tol))


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit exact low-N finite-volume microstate counts for g_N.")
    ap.add_argument(
        "--profile-csv",
        type=Path,
        default=DEFAULT_OUTDIR / "gn_phase_space_2d_Dgrid21.csv",
        help="Low-N phase-space profile CSV.",
    )
    ap.add_argument(
        "--spectrum-csv",
        type=Path,
        default=DEFAULT_OUTDIR / "gn_phase_space_2d_spectrum_Dgrid21.csv",
        help="Low-mode spectral CSV from the same extraction run.",
    )
    ap.add_argument("--level", default="fine", help="Grid level to audit.")
    ap.add_argument("--count-tol", type=float, default=1e-10, help="Tolerance for spectral counting.")
    ap.add_argument("--gap-tol", type=float, default=1e-8, help="Minimum accepted strict gap.")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    prof = pd.read_csv(args.profile_csv)
    spec = pd.read_csv(args.spectrum_csv)
    prof = prof[prof["level"].astype(str) == str(args.level)].copy()
    spec = spec[spec["level"].astype(str) == str(args.level)].copy()
    if prof.empty:
        raise RuntimeError(f"No profile rows found for level={args.level!r} in {args.profile_csv}.")
    if spec.empty:
        raise RuntimeError(f"No spectrum rows found for level={args.level!r} in {args.spectrum_csv}.")

    detail_rows: list[dict[str, Any]] = []
    for _, prow in prof.sort_values("D").iterrows():
        D = float(prow["D"])
        sD = spec[np.isclose(spec["D"].astype(float), D)].copy()
        if len(sD) < 3:
            raise RuntimeError(f"D={D}: need at least three spectral rows, got {len(sD)}.")

        sD = sD.sort_values(["lambda_n", "mode_n"]).reset_index(drop=True)
        spectral_lambdas = sD["lambda_n"].to_numpy(dtype=float)
        lam = [float(prow[f"lambda{i}"]) for i in (1, 2, 3)]
        spec_lam = [float(spectral_lambdas[i - 1]) for i in (1, 2, 3)]
        gaps = [spec_lam[1] - spec_lam[0], spec_lam[2] - spec_lam[1]]
        ranks = [finite_count(spectral_lambdas, spec_lam[i - 1], args.count_tol) for i in (1, 2, 3)]
        rank_errors = [abs(ranks[i - 1] - i) for i in (1, 2, 3)]
        exact_ratios = [1.0 / 3.0, 2.0 / 3.0, 1.0]
        ghat = [float(prow[f"g{i}_hat"]) for i in (1, 2, 3)]
        ratio_delta = [abs(exact_ratios[i] - ghat[i]) for i in range(3)]
        lambda_profile_spectrum_max_abs = max(abs(lam[i] - spec_lam[i]) for i in range(3))
        strict_gap_pass = min(gaps) > args.gap_tol
        rank_pass = max(rank_errors) == 0

        detail_rows.append(
            {
                "D": D,
                "level": args.level,
                "lambda1": spec_lam[0],
                "lambda2": spec_lam[1],
                "lambda3": spec_lam[2],
                "profile_spectrum_lambda_max_abs": lambda_profile_spectrum_max_abs,
                "gap12": gaps[0],
                "gap23": gaps[1],
                "rank_at_lambda1": ranks[0],
                "rank_at_lambda2": ranks[1],
                "rank_at_lambda3": ranks[2],
                "max_rank_error": max(rank_errors),
                "exact_ratio1": exact_ratios[0],
                "exact_ratio2": exact_ratios[1],
                "exact_ratio3": exact_ratios[2],
                "shell_volume_g1_hat": ghat[0],
                "shell_volume_g2_hat": ghat[1],
                "shell_volume_g3_hat": ghat[2],
                "abs_exact_minus_shell_g1": ratio_delta[0],
                "abs_exact_minus_shell_g2": ratio_delta[1],
                "abs_exact_minus_shell_g3": ratio_delta[2],
                "strict_gap_pass": bool(strict_gap_pass),
                "rank_count_pass": bool(rank_pass),
                "row_pass": bool(strict_gap_pass and rank_pass),
            }
        )

    detail = pd.DataFrame(detail_rows)
    all_pass = bool(detail["row_pass"].all())
    summary_rows = [
        {
            "level": args.level,
            "n_D": int(len(detail)),
            "min_gap12": float(detail["gap12"].min()),
            "min_gap23": float(detail["gap23"].min()),
            "max_profile_spectrum_lambda_abs": float(detail["profile_spectrum_lambda_max_abs"].max()),
            "max_rank_error": int(detail["max_rank_error"].max()),
            "g1_hat_min": float(detail["shell_volume_g1_hat"].min()),
            "g1_hat_max": float(detail["shell_volume_g1_hat"].max()),
            "g2_hat_min": float(detail["shell_volume_g2_hat"].min()),
            "g2_hat_max": float(detail["shell_volume_g2_hat"].max()),
            "exact_ratio1": 1.0 / 3.0,
            "exact_ratio2": 2.0 / 3.0,
            "exact_ratio3": 1.0,
            "max_exact_ratio_delta_g1": float(detail["abs_exact_minus_shell_g1"].max()),
            "max_exact_ratio_delta_g2": float(detail["abs_exact_minus_shell_g2"].max()),
            "max_exact_ratio_delta_g3": float(detail["abs_exact_minus_shell_g3"].max()),
            "all_strict_gap_pass": bool(detail["strict_gap_pass"].all()),
            "all_rank_count_pass": bool(detail["rank_count_pass"].all()),
            "all_pass": all_pass,
        }
    ]

    detail_path = args.outdir / "gn_lowN_microstate_count_detail.csv"
    summary_path = args.outdir / "gn_lowN_microstate_count_summary.csv"
    write_rows(detail_path, detail_rows)
    write_rows(summary_path, summary_rows)

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(detail_path, PAPER_DIR / detail_path.name)
    shutil.copy2(summary_path, PAPER_DIR / summary_path.name)

    if not all_pass:
        raise SystemExit("low-N microstate count audit failed; see detail CSV.")

    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"All finite-volume count checks passed for level={args.level}, n_D={len(detail_rows)}.")


if __name__ == "__main__":
    main()
