#!/usr/bin/env python3
"""
Quantify global-scan impact of first-principles g_N candidates.

Cases:
  1) baseline: g_mode = fp_2d_full
  2) legacy surrogate reference: g_mode = cardy
  3) first-principles 1D candidate: g_mode = fp_1d
  4) first-principles 2D candidate: g_mode = fp_2d

Outputs:
  - output/gn_fp_impact/gn_profile_impact.csv
  - output/gn_fp_impact/gn_profile_values.csv
  - paper/gn_profile_impact.csv
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((ROOT / "code").resolve()))

from pslt_lib import PSLTKinetics, PSLTParameters


OUTDIR = ROOT / "output" / "gn_fp_impact"
PAPER_DIR = ROOT / "paper"
G_FP_BLEND = 0.01


@dataclass(frozen=True)
class Case:
    name: str
    g_mode: str


def make_kinetics(g_mode: str) -> PSLTKinetics:
    params = PSLTParameters(
        c_eff=0.5,
        nu=5.0,
        kappa_g=0.03,
        g_mode=g_mode,
        g_fp_blend=G_FP_BLEND,
        chi=0.2,
        chi_mode="localized_interp",
        chi_lr_D=(6.0, 12.0, 18.0),
        chi_lr_vals=(4.01827e-4, 2.21414e-4, 2.13187e-4),
        A1=1.0,
        A2=1.0,
        b_n_power=0.30,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
    )
    return PSLTKinetics(params)


def evaluate_case(case: Case) -> Dict[str, float]:
    kin = make_kinetics(case.g_mode)

    D_vals = np.linspace(4.0, 20.0, 60)
    eta_vals = np.linspace(0.2, 4.0, 60)
    t_coh = 1.0
    N_max = 20

    mu_obs = 1.4
    sigma_obs = 0.4
    D0, eta0 = 10.0, 1.0

    def W2(D: float, eta: float) -> float:
        N = 2
        gam = kin.calculate_gamma_N(N, D, eta)
        g = kin.g_N_effective(N, D)
        B = kin.B_N(N)
        return float(B * g * (1.0 - np.exp(-gam * t_coh)))

    W2_ref = W2(D0, eta0)
    if W2_ref <= 0:
        raise RuntimeError(f"{case.name}: non-positive W2_ref at baseline point.")

    r3_vals: List[float] = []
    winner_vals: List[int] = []
    tail_vals: List[float] = []
    chi2_vals: List[float] = []

    for eta in eta_vals:
        for D in D_vals:
            _, _, meta = kin.get_probabilities(D, eta, t_coh, N_max=N_max)
            r3 = float(meta["generation_ratio"])
            tail = float(max(1.0 - r3, 0.0))
            winner = int(meta["winner"])

            mu_pred = W2(D, eta) / W2_ref
            chi2 = float(((mu_pred - mu_obs) / sigma_obs) ** 2)

            r3_vals.append(r3)
            tail_vals.append(tail)
            winner_vals.append(winner)
            chi2_vals.append(chi2)

    r3_arr = np.asarray(r3_vals, dtype=float)
    tail_arr = np.asarray(tail_vals, dtype=float)
    win_arr = np.asarray(winner_vals, dtype=int)
    chi2_arr = np.asarray(chi2_vals, dtype=float)

    return {
        "case": case.name,
        "g_mode_requested": case.g_mode,
        "g_mode_active": kin.active_g_mode(),
        "g_fp_blend": G_FP_BLEND,
        "f_R3_gt_0p90": float(np.mean(r3_arr >= 0.90)),
        "f_R3_gt_0p95": float(np.mean(r3_arr >= 0.95)),
        "f_hmumu_chi2_le_4": float(np.mean(chi2_arr <= 4.0)),
        "best_chi2": float(np.min(chi2_arr)),
        "f_winner_gt_3": float(np.mean(win_arr > 3)),
        "mean_tail_prob": float(np.mean(tail_arr)),
        "max_tail_prob": float(np.max(tail_arr)),
        "f_tail_gt_0p10": float(np.mean(tail_arr > 0.10)),
        "g1_D12": float(kin.g_N_effective(1, 12.0)),
        "g2_D12": float(kin.g_N_effective(2, 12.0)),
        "g3_D12": float(kin.g_N_effective(3, 12.0)),
    }


def export_profile_samples(cases: List[Case]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    D_grid = np.linspace(4.0, 20.0, 17)
    for case in cases:
        kin = make_kinetics(case.g_mode)
        for D in D_grid:
            rows.append(
                {
                    "case": case.name,
                    "g_mode_requested": case.g_mode,
                    "g_mode_active": kin.active_g_mode(),
                    "g_fp_blend": G_FP_BLEND,
                    "D": float(D),
                    "g1": float(kin.g_N_effective(1, D)),
                    "g2": float(kin.g_N_effective(2, D)),
                    "g3": float(kin.g_N_effective(3, D)),
                    "g4": float(kin.g_N_effective(4, D)),
                    "g5": float(kin.g_N_effective(5, D)),
                    "g6": float(kin.g_N_effective(6, D)),
                }
            )
    return rows


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        Case(name="baseline_fp_2d_full", g_mode="fp_2d_full"),
        Case(name="legacy_cardy", g_mode="cardy"),
        Case(name="first_principles_fp_1d", g_mode="fp_1d"),
        Case(name="first_principles_fp_2d", g_mode="fp_2d"),
    ]

    rows = [evaluate_case(c) for c in cases]
    out_csv = OUTDIR / "gn_profile_impact.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    profile_rows = export_profile_samples(cases)
    out_profile = OUTDIR / "gn_profile_values.csv"
    with out_profile.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(profile_rows)

    paper_csv = PAPER_DIR / "gn_profile_impact.csv"
    paper_csv.write_text(out_csv.read_text())

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_profile}")
    print(f"[saved] {paper_csv}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
