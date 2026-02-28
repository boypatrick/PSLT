#!/usr/bin/env python3
"""
Calibrate UV-tree observable knobs against the matched baseline.

Target metric:
  f(chi2_mumu <= 4) under observable_mode="eft_wilson_uv_tree"
close to the matched-baseline value on the same map grid.

Outputs:
  - output/hll_signal_strength/hll_uv_tree_calibration.csv
  - output/hll_signal_strength/hll_uv_tree_calibration_best.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((ROOT / "code").resolve()))

from action_grid_profile_utils import scan_d_values, select_chi_profile, select_superrad_profile  # noqa: E402
from hll_observable import HLLChannelPredictor, HLLObservableConfig  # noqa: E402
from pslt_lib import PSLTKinetics, PSLTParameters  # noqa: E402


OUTDIR = ROOT / "output" / "hll_signal_strength"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"

D_MIN, D_MAX, D_NUM = 4.0, 20.0, 60
ETA_MIN, ETA_MAX, ETA_NUM = 0.2, 4.0, 60
REF_D, REF_ETA = 10.0, 1.0
T_COH = 1.0
N_MAX = 20
MU_OBS = 1.4
SIGMA_OBS = 0.4

BLEND_GRID = [0.00, 0.15, 0.25, 0.35, 0.50, 0.70, 1.00]
M2_POWER_GRID = [0.00, 0.50, 0.75, 1.00, 1.25]


def make_kinetics(observable_mode: str, uv_blend: float, uv_m2_power: float) -> PSLTKinetics:
    d_scan = scan_d_values(D_MIN, D_MAX, D_NUM)
    chi_prof = select_chi_profile(ROOT, d_scan)
    superrad_prof = select_superrad_profile(ROOT, d_scan)
    params = PSLTParameters(
        c_eff=0.5,
        nu=5.0,
        kappa_g=0.03,
        g_mode="fp_2d_full",
        g_fp_norm_mode="phase_space",
        g_fp_full_window_blend=0.8,
        g_fp_full_tail_beta=1.1,
        g_fp_full_tail_shell_power=0.0,
        g_fp_full_tail_clip_min=1e-3,
        g_fp_full_tail_clip_max=0.95,
        chi=0.2,
        chi_mode=str(chi_prof["mode"]),
        chi_lr_D=tuple(float(x) for x in chi_prof["d"]),
        chi_lr_vals=tuple(float(x) for x in chi_prof["chi"]),
        A1=1.0,
        A2=1.0,
        gamma_mode=str(superrad_prof["mode"]),
        gamma_superrad_csv=str(superrad_prof["path"]),
        b_mode="eft_operator_norm",
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=0.30,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode=str(observable_mode),
        hll_observable_nmax=N_MAX,
        hll_uv_blend=float(uv_blend),
        hll_uv_m2_power=float(uv_m2_power),
    )
    return PSLTKinetics(params)


def evaluate_mumu(kin: PSLTKinetics, observable_mode: str) -> Dict[str, float]:
    d_vals = np.linspace(D_MIN, D_MAX, D_NUM)
    eta_vals = np.linspace(ETA_MIN, ETA_MAX, ETA_NUM)
    cfg = HLLObservableConfig(
        mode=str(observable_mode),
        t_coh=T_COH,
        ref_D=REF_D,
        ref_eta=REF_ETA,
        n_max=N_MAX,
    )
    pred = HLLChannelPredictor(kin, layer_n=2, cfg=cfg)

    mu_vals = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            mu_vals[i, j] = float(pred.mu_pred(float(d), float(eta)))

    chi2 = ((mu_vals - MU_OBS) / SIGMA_OBS) ** 2
    return {
        "f_chi2_le_4": float(np.mean(chi2 <= 4.0)),
        "best_chi2": float(np.min(chi2)),
        "mu_p50": float(np.percentile(mu_vals, 50)),
        "mu_p90": float(np.percentile(mu_vals, 90)),
    }


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    kin_match = make_kinetics("eft_wilson_matched", uv_blend=1.0, uv_m2_power=1.0)
    baseline = evaluate_mumu(kin_match, "eft_wilson_matched")
    target = float(baseline["f_chi2_le_4"])
    print(f"[target] matched f(chi2<=4) = {target:.6f}")

    rows: List[Dict[str, float]] = []
    for blend in BLEND_GRID:
        for pwr in M2_POWER_GRID:
            kin_uv = make_kinetics("eft_wilson_uv_tree", uv_blend=float(blend), uv_m2_power=float(pwr))
            stats = evaluate_mumu(kin_uv, "eft_wilson_uv_tree")
            drift = float(stats["f_chi2_le_4"] - target)
            score = abs(drift)
            row = {
                "uv_blend": float(blend),
                "uv_m2_power": float(pwr),
                "f_chi2_le_4": float(stats["f_chi2_le_4"]),
                "best_chi2": float(stats["best_chi2"]),
                "mu_p50": float(stats["mu_p50"]),
                "mu_p90": float(stats["mu_p90"]),
                "target_f_chi2_le_4": float(target),
                "delta_f_chi2_le_4": drift,
                "score_abs_delta": float(score),
            }
            rows.append(row)
            print(
                f"[scan] blend={blend:.2f}, p={pwr:.2f}, "
                f"f={row['f_chi2_le_4']:.4f}, delta={drift:+.4f}"
            )

    rows_sorted = sorted(rows, key=lambda r: (r["score_abs_delta"], r["best_chi2"]))
    best = rows_sorted[0]

    out_csv = OUTDIR / "hll_uv_tree_calibration.csv"
    out_json = OUTDIR / "hll_uv_tree_calibration_best.json"
    write_csv(out_csv, rows_sorted)
    out_json.write_text(json.dumps(best, indent=2))

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")
    print("[best]", best)


if __name__ == "__main__":
    main()
