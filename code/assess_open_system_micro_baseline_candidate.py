#!/usr/bin/env python3
"""
Assess open_system_micro as a baseline-candidate (go/no-go gates).

This is a decision layer, not a physics extractor. It consumes reproducible
CSV outputs and evaluates explicit promotion gates.

Inputs (defaults):
  - output/gn_fp_impact/gn_profile_impact.csv
  - output/chi_open_system/chi_open_system_micro_sensitivity.csv
  - output/chi_open_system/kappa_env_anchor_calibration.csv
  - output/chi_open_system/kappa_env_anchor_holdout.csv

Outputs:
  - output/chi_open_system/open_system_micro_baseline_candidate.csv
  - output/chi_open_system/open_system_micro_baseline_candidate.json
  - paper/open_system_micro_baseline_candidate.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"

BASELINE_GN_CSV = ROOT / "output" / "gn_fp_impact" / "gn_profile_impact.csv"
MICRO_SENS_CSV = ROOT / "output" / "chi_open_system" / "chi_open_system_micro_sensitivity.csv"
CALIB_CSV = ROOT / "output" / "chi_open_system" / "kappa_env_anchor_calibration.csv"
HOLDOUT_CSV = ROOT / "output" / "chi_open_system" / "kappa_env_anchor_holdout.csv"


def _pick_row(df: pd.DataFrame, case: str) -> pd.Series:
    sub = df[df["case"] == case]
    if sub.empty:
        raise KeyError(f"case='{case}' not found in CSV.")
    return sub.iloc[0]


def _b(value: bool) -> int:
    return 1 if bool(value) else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Go/No-Go gate assessment for open_system_micro baseline-candidate.")
    ap.add_argument("--baseline-csv", default=str(BASELINE_GN_CSV))
    ap.add_argument("--micro-csv", default=str(MICRO_SENS_CSV))
    ap.add_argument("--calib-csv", default=str(CALIB_CSV))
    ap.add_argument("--holdout-csv", default=str(HOLDOUT_CSV))

    # Gates
    ap.add_argument("--thr-delta-r3", type=float, default=0.01)
    ap.add_argument("--thr-delta-accept", type=float, default=0.01)
    ap.add_argument("--thr-delta-winner-gt3", type=float, default=5e-4)
    ap.add_argument("--thr-ratio-dynamic-range", type=float, default=0.10)
    ap.add_argument("--thr-holdout-rmse", type=float, default=0.08)
    ap.add_argument("--thr-holdout-max-abs", type=float, default=0.18)
    args = ap.parse_args()

    base_df = pd.read_csv(args.baseline_csv)
    micro_df = pd.read_csv(args.micro_csv)
    calib_df = pd.read_csv(args.calib_csv)
    holdout_df = pd.read_csv(args.holdout_csv)

    base = _pick_row(base_df, "baseline_fp_2d_full")
    micro = _pick_row(micro_df, "open_micro_base")
    calib = calib_df.iloc[0]
    holdout = holdout_df.iloc[0]

    f_r3_base = float(base["f_R3_gt_0p90"])
    f_acc_base = float(base["f_hmumu_chi2_le_4"])
    f_win_base = float(base["f_winner_gt_3"])

    f_r3_micro = float(micro["f_R3_gt_0p90"])
    f_acc_micro = float(micro["f_hmumu_chi2_le_4"])
    f_win_micro = float(micro["f_winner_gt_3"])

    d_r3 = float(f_r3_micro - f_r3_base)
    d_acc = float(f_acc_micro - f_acc_base)
    d_win = float(f_win_micro - f_win_base)

    ratio_min = float(micro["ratio_min"])
    ratio_max = float(micro["ratio_max"])
    ratio_mean = float(micro["ratio_mean"])
    ratio_rng = float(ratio_max - ratio_min)

    g_anchor_rule_single = _b(
        "target_definition" in calib.index
        and "single-point anchor" in str(calib["target_definition"]).lower()
    )
    g_anchor_rule_multi = _b(
        ("anchor_mode" in calib.index and "multi_anchor" in str(calib["anchor_mode"]).lower())
        or (
            "target_definition" in calib.index
            and "multi-anchor" in str(calib["target_definition"]).lower()
        )
    )

    holdout_rmse = float(holdout["holdout_rmse"]) if "holdout_rmse" in holdout.index else float("nan")
    holdout_max_abs = float(holdout["holdout_max_abs_err"]) if "holdout_max_abs_err" in holdout.index else float("nan")
    n_holdout = int(holdout["n_holdout"]) if "n_holdout" in holdout.index else 0
    g_holdout = _b(
        np.isfinite(holdout_rmse)
        and np.isfinite(holdout_max_abs)
        and n_holdout > 0
        and holdout_rmse <= float(args.thr_holdout_rmse)
        and holdout_max_abs <= float(args.thr_holdout_max_abs)
    )

    g_r3 = _b(abs(d_r3) <= float(args.thr_delta_r3))
    g_acc = _b(abs(d_acc) <= float(args.thr_delta_accept))
    g_winner = _b(abs(d_win) <= float(args.thr_delta_winner_gt3))
    g_dyn = _b(ratio_rng >= float(args.thr_ratio_dynamic_range))

    go = _b(bool(g_anchor_rule_multi and g_holdout and g_r3 and g_acc and g_winner and g_dyn))

    out: Dict[str, float | int | str] = {
        "candidate_mode": "open_system_micro",
        "decision": "GO_baseline_candidate" if go else "NO_GO_keep_diagnostic",
        "go_flag": go,
        "gate_anchor_rule_single_point": g_anchor_rule_single,
        "gate_anchor_rule_multi_anchor": g_anchor_rule_multi,
        "gate_holdout_validation": g_holdout,
        "gate_stability_R3": g_r3,
        "gate_stability_acceptance": g_acc,
        "gate_stability_winner_gt3": g_winner,
        "gate_nontrivial_dynamic_range": g_dyn,
        "delta_f_R3_gt_0p90": d_r3,
        "delta_f_hmumu_chi2_le_4": d_acc,
        "delta_f_winner_gt_3": d_win,
        "f_R3_gt_0p90_baseline": f_r3_base,
        "f_R3_gt_0p90_micro": f_r3_micro,
        "f_hmumu_chi2_le_4_baseline": f_acc_base,
        "f_hmumu_chi2_le_4_micro": f_acc_micro,
        "f_winner_gt_3_baseline": f_win_base,
        "f_winner_gt_3_micro": f_win_micro,
        "ratio_min_micro": ratio_min,
        "ratio_max_micro": ratio_max,
        "ratio_mean_micro": ratio_mean,
        "ratio_dynamic_range_micro": ratio_rng,
        "kappa_env_calibrated": float(calib["kappa_env_calibrated"]),
        "kappa_anchor_D_ref": float(calib["D_ref"]) if "D_ref" in calib.index else float("nan"),
        "kappa_anchor_ratio_target": float(calib["ratio_anchor_target"]),
        "kappa_anchor_ratio_pred": float(calib["ratio_anchor_pred"]),
        "kappa_anchor_abs_err": float(calib["ratio_anchor_abs_err"]),
        "kappa_anchor_n_anchor": int(calib["n_anchor"]) if "n_anchor" in calib.index else 0,
        "kappa_anchor_n_holdout": n_holdout,
        "kappa_anchor_holdout_rmse": holdout_rmse,
        "kappa_anchor_holdout_max_abs_err": holdout_max_abs,
        "threshold_delta_r3": float(args.thr_delta_r3),
        "threshold_delta_accept": float(args.thr_delta_accept),
        "threshold_delta_winner_gt3": float(args.thr_delta_winner_gt3),
        "threshold_ratio_dynamic_range": float(args.thr_ratio_dynamic_range),
        "threshold_holdout_rmse": float(args.thr_holdout_rmse),
        "threshold_holdout_max_abs_err": float(args.thr_holdout_max_abs),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTDIR / "open_system_micro_baseline_candidate.csv"
    out_json = OUTDIR / "open_system_micro_baseline_candidate.json"
    paper_csv = PAPER_DIR / "open_system_micro_baseline_candidate.csv"

    pd.DataFrame([out]).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(out, indent=2))
    paper_csv.write_text(out_csv.read_text())

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")
    print(f"[saved] {paper_csv}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
