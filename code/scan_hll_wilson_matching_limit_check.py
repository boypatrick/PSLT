#!/usr/bin/env python3
"""
Consistency audit for H->ll matching mode.

Goal:
  Verify that the new mode `eft_wilson_matched` reproduces the legacy
  `eft_wilson_diag` result in the strict diagonal limit:
    mix_scale = 0, width_mode = none, basis_mode = sqrt_yraw.

Outputs:
  - output/hll_matching/matched_limit_check_points.csv
  - output/hll_matching/matched_limit_check_summary.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from pslt_lib import PSLTKinetics, PSLTParameters
from hll_observable import HLLObservableConfig, HLLChannelPredictor


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "hll_matching"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"


def make_kinetics(mode: str) -> PSLTKinetics:
    return PSLTKinetics(
        PSLTParameters(
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
            chi_mode="localized_interp",
            chi_lr_D=(6.0, 12.0, 18.0),
            chi_lr_vals=(4.01827e-4, 2.21414e-4, 2.13187e-4),
            A1=1.0,
            A2=1.0,
            gamma_mode="action_profile",
            b_mode="overlap_2d",
            b_overlap_csv=str(B_OVERLAP_CSV),
            b_n_power=0.30,
            b_n_mode="cumulative",
            b_n_tail_mode="saturate",
            hll_observable_mode=mode,
            hll_observable_nmax=20,
            hll_match_basis_mode="sqrt_yraw",
            hll_match_mix_scale=0.0,
            hll_match_mix_max=0.25,
            hll_match_eta_power=1.0,
            hll_match_eta_ref=1.0,
            hll_match_width_mode="none",
            hll_match_width_scale=0.0,
        )
    )


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    kin_diag = make_kinetics("eft_wilson_diag")
    kin_match = make_kinetics("eft_wilson_matched")

    cfg_diag = HLLObservableConfig(mode="eft_wilson_diag", t_coh=1.0, ref_D=10.0, ref_eta=1.0, n_max=20)
    cfg_match = HLLObservableConfig(mode="eft_wilson_matched", t_coh=1.0, ref_D=10.0, ref_eta=1.0, n_max=20)
    pred_diag = HLLChannelPredictor(kin_diag, layer_n=2, cfg=cfg_diag)
    pred_match = HLLChannelPredictor(kin_match, layer_n=2, cfg=cfg_match)

    d_vals = [6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
    eta_vals = [0.5, 1.0, 2.0, 3.0]
    rows: List[Dict[str, float]] = []
    abs_dev = []
    rel_dev = []
    rel_dev_masked = []
    for d in d_vals:
        for eta in eta_vals:
            mu_d = float(pred_diag.mu_pred(d, eta))
            mu_m = float(pred_match.mu_pred(d, eta))
            delta = mu_m - mu_d
            rel = abs(delta) / max(abs(mu_d), 1e-30)
            if abs(mu_d) > 1e-6:
                rel_dev_masked.append(rel)
            abs_dev.append(abs(delta))
            rel_dev.append(rel)
            rows.append(
                {
                    "D": d,
                    "eta": eta,
                    "mu_diag": mu_d,
                    "mu_matched_limit": mu_m,
                    "delta_mu": delta,
                    "rel_delta_mu": rel,
                }
            )

    summary = [
        {
            "n_points": len(rows),
            "max_abs_delta_mu": float(np.max(abs_dev)),
            "max_rel_delta_mu": float(np.max(rel_dev)),
            "max_rel_delta_mu_mu_gt_1e-6": float(np.max(rel_dev_masked) if rel_dev_masked else 0.0),
            "mean_abs_delta_mu": float(np.mean(abs_dev)),
            "mean_rel_delta_mu": float(np.mean(rel_dev)),
            "mean_rel_delta_mu_mu_gt_1e-6": float(np.mean(rel_dev_masked) if rel_dev_masked else 0.0),
        }
    ]

    out_points = OUTDIR / "matched_limit_check_points.csv"
    out_summary = OUTDIR / "matched_limit_check_summary.csv"
    write_csv(out_points, rows)
    write_csv(out_summary, summary)
    print(f"[saved] {out_points}")
    print(f"[saved] {out_summary}")
    print(summary[0])


if __name__ == "__main__":
    main()
