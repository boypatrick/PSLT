#!/usr/bin/env python3
"""
Minimal conformal Dirac-Higgs frame check used by the PSLT visibility baseline.

Checks:
  1) Dirac kinetic conformal power cancellation with Psi = Omega^{-3/2}.
  2) Yukawa conformal power cancellation with H = Omega^{-1}.
  3) Optional profile CSV metadata consistency (frame_power, frame model).
  4) Common spin-degeneracy bookkeeping does not change normalized layer
     probabilities or profile ratios.

Outputs:
  - output/dirac_frame_check/minimal_dirac_conformal_check.csv
  - paper/minimal_dirac_conformal_check.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "dirac_frame_check"
PAPER_DIR = ROOT / "paper"


def to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Check minimal Dirac conformal frame cancellation.")
    ap.add_argument(
        "--profile-csv",
        type=Path,
        default=ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv",
        help="Optional overlap-profile CSV to validate baseline frame metadata.",
    )
    ap.add_argument(
        "--probability-csv",
        type=Path,
        default=ROOT / "paper" / "surrogate_vs_action_points.csv",
        help="Optional point-probability CSV used to audit common spin-factor invariance.",
    )
    ap.add_argument(
        "--spin-factor",
        type=float,
        default=2.0,
        help="Positive common spin degeneracy factor used in the invariance audit.",
    )
    args = ap.parse_args()
    if args.spin_factor <= 0:
        raise ValueError("--spin-factor must be positive.")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    # Conformal powers: sqrt(-g)=Omega^4, gamma^mu=e_a^mu gamma^a with e_a^mu~Omega^-1.
    p_sqrtg = 4.0
    p_gamma = -1.0
    p_dirac = -1.5
    p_higgs = -1.0

    kinetic_net = p_sqrtg + 2.0 * p_dirac + p_gamma
    yukawa_net = p_sqrtg + p_dirac + p_higgs + p_dirac
    d_spin = float(args.spin_factor)

    rows: list[dict[str, Any]] = [
        {
            "check": "kinetic_conformal_power",
            "expression": "4 + 2*p_dirac - 1",
            "value": kinetic_net,
            "target": 0.0,
            "pass": abs(kinetic_net) < 1e-12,
        },
        {
            "check": "yukawa_conformal_power",
            "expression": "4 + p_dirac + p_higgs + p_dirac",
            "value": yukawa_net,
            "target": 0.0,
            "pass": abs(yukawa_net) < 1e-12,
        },
    ]

    if args.profile_csv.exists():
        df = pd.read_csv(args.profile_csv)
        n_points = int(len(df))

        frame_power_vals = []
        if "frame_power" in df.columns:
            frame_power_vals = sorted({float(x) for x in df["frame_power"].dropna().to_list()})

        frame_model_vals = []
        if "frame_model" in df.columns:
            frame_model_vals = sorted({str(x) for x in df["frame_model"].dropna().to_list()})

        ypow_vals = []
        if "yukawa_net_omega_power" in df.columns:
            ypow_vals = [to_float(x) for x in df["yukawa_net_omega_power"].dropna().to_list()]

        frame_power_ok = (len(frame_power_vals) == 1 and abs(frame_power_vals[0]) < 1e-12) if frame_power_vals else True
        frame_model_ok = (len(frame_model_vals) == 1 and frame_model_vals[0] == "minimal_dirac_conformal") if frame_model_vals else True
        ypow_ok = (len(ypow_vals) > 0 and float(np.nanmax(np.abs(np.asarray(ypow_vals, dtype=float)))) < 1e-12) if ypow_vals else True

        rows.extend(
            [
                {
                    "check": "profile_frame_power_baseline",
                    "expression": "unique(frame_power)",
                    "value": "|".join(f"{x:.6g}" for x in frame_power_vals) if frame_power_vals else "not_available",
                    "target": "0 (if column exists)",
                    "pass": bool(frame_power_ok),
                },
                {
                    "check": "profile_frame_model_tag",
                    "expression": "unique(frame_model)",
                    "value": "|".join(frame_model_vals) if frame_model_vals else "not_available",
                    "target": "minimal_dirac_conformal (if column exists)",
                    "pass": bool(frame_model_ok),
                },
                {
                    "check": "profile_yukawa_net_power",
                    "expression": "max_abs(yukawa_net_omega_power)",
                    "value": float(np.nanmax(np.abs(np.asarray(ypow_vals, dtype=float)))) if ypow_vals else "not_available",
                    "target": "0 (if column exists)",
                    "pass": bool(ypow_ok),
                },
                {
                    "check": "profile_points",
                    "expression": "len(profile_csv)",
                    "value": n_points,
                    "target": ">=1",
                    "pass": n_points >= 1,
                },
            ]
        )

        raw_cols = [f"y_eff_raw_{i}" for i in (1, 2, 3)]
        if all(c in df.columns for c in raw_cols):
            raw = df[raw_cols].to_numpy(dtype=float)
            raw_norm = raw / np.maximum(np.sum(raw, axis=1, keepdims=True), 1e-300)
            raw_lifted = (d_spin * raw) / np.maximum(np.sum(d_spin * raw, axis=1, keepdims=True), 1e-300)
            raw_resid = float(np.nanmax(np.abs(raw_norm - raw_lifted)))
            rows.append(
                {
                    "check": "profile_raw_spin_factor_invariance",
                    "expression": "normalize(d_spin*y_eff_raw)-normalize(y_eff_raw)",
                    "value": raw_resid,
                    "target": "<1e-14",
                    "pass": raw_resid < 1e-14,
                }
            )

        cum_cols = [f"y_eff_cum_{i}" for i in (1, 2, 3)]
        if all(c in df.columns for c in cum_cols):
            cum = df[cum_cols].to_numpy(dtype=float)
            base_b = cum / np.maximum(cum[:, [2]], 1e-300)
            lifted_b = (d_spin * cum) / np.maximum(d_spin * cum[:, [2]], 1e-300)
            b_resid = float(np.nanmax(np.abs(base_b - lifted_b)))
            rows.append(
                {
                    "check": "profile_cumulative_spin_factor_invariance",
                    "expression": "(d_spin*y_eff_cum)/(d_spin*y_eff_cum_3)-y_eff_cum/y_eff_cum_3",
                    "value": b_resid,
                    "target": "<1e-14",
                    "pass": b_resid < 1e-14,
                }
            )
    else:
        rows.append(
            {
                "check": "profile_csv_presence",
                "expression": str(args.profile_csv),
                "value": "missing",
                "target": "exists",
                "pass": False,
            }
        )

    if args.probability_csv.exists():
        pdf = pd.read_csv(args.probability_csv)
        resid_max = 0.0
        winner_ok = True
        n_prob_rows = int(len(pdf))
        n_blocks = 0
        for suffix in ("surrogate", "direct"):
            prob_cols = [f"P{i}_{suffix}" for i in (1, 2, 3)]
            if not all(c in pdf.columns for c in prob_cols):
                continue
            probs3 = pdf[prob_cols].to_numpy(dtype=float)
            tail = np.maximum(1.0 - np.sum(probs3, axis=1, keepdims=True), 0.0)
            probs = np.concatenate([probs3, tail], axis=1)
            lifted = d_spin * probs
            lifted /= np.maximum(np.sum(lifted, axis=1, keepdims=True), 1e-300)
            resid_max = max(resid_max, float(np.nanmax(np.abs(probs - lifted))))
            winner_ok = winner_ok and bool(np.all(np.argmax(probs3, axis=1) == np.argmax(lifted[:, :3], axis=1)))
            n_blocks += 1

        rows.extend(
            [
                {
                    "check": "spin_factor_value",
                    "expression": "d_spin",
                    "value": d_spin,
                    "target": ">0 and layer-independent",
                    "pass": d_spin > 0.0,
                },
                {
                    "check": "probability_spin_factor_invariance",
                    "expression": "normalize(d_spin*[P1,P2,P3,Ptail])-normalize([P1,P2,P3,Ptail])",
                    "value": resid_max if n_blocks else "not_available",
                    "target": "<1e-14",
                    "pass": bool(n_blocks and resid_max < 1e-14),
                },
                {
                    "check": "winner_spin_factor_invariance",
                    "expression": "argmax(P1,P2,P3) after common spin lift",
                    "value": f"{n_prob_rows} rows, {n_blocks} probability blocks",
                    "target": "unchanged",
                    "pass": bool(n_blocks and winner_ok),
                },
            ]
        )

    out_csv = OUTDIR / "minimal_dirac_conformal_check.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["check", "expression", "value", "target", "pass"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    paper_csv = PAPER_DIR / "minimal_dirac_conformal_check.csv"
    paper_csv.write_bytes(out_csv.read_bytes())

    all_pass = all(bool(r["pass"]) for r in rows)
    print(f"[saved] {out_csv}")
    print(f"[saved] {paper_csv}")
    print(f"[summary] all_pass={all_pass}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
