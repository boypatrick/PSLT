#!/usr/bin/env python3
"""
N_max convergence scan for g_N migration tests.

This script evaluates map-level metrics at multiple N_max values for:
  - baseline_fp_2d_full (g_mode=fp_2d_full)
  - first_principles_fp_2d (g_mode=fp_2d)

Outputs:
  - output/gn_fp_impact/gn_nmax_convergence.csv
  - paper/gn_nmax_convergence.csv
"""

from __future__ import annotations

import argparse
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
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"


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
        b_mode="overlap_2d",
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=0.30,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
    )
    return PSLTKinetics(params)


def eval_case_nmax(case: Case, n_max: int) -> Dict[str, float]:
    kin = make_kinetics(case.g_mode)
    D_vals = np.linspace(4.0, 20.0, 60)
    eta_vals = np.linspace(0.2, 4.0, 60)
    t_coh = 1.0
    mu_obs = 1.4
    sigma_obs = 0.4
    D0, eta0 = 10.0, 1.0

    def W2(D: float, eta: float) -> float:
        N = 2
        gam = kin.calculate_gamma_N(N, D, eta)
        g = kin.g_N_effective(N, D)
        B = kin.B_N(N, D)
        return float(B * g * (1.0 - np.exp(-gam * t_coh)))

    W2_ref = W2(D0, eta0)
    if W2_ref <= 0:
        raise RuntimeError(f"{case.name}: non-positive W2_ref at baseline point.")

    r3_list: List[float] = []
    winner_list: List[int] = []
    tail_list: List[float] = []
    chi2_list: List[float] = []

    for eta in eta_vals:
        for D in D_vals:
            _, _, meta = kin.get_probabilities(float(D), float(eta), t_coh, N_max=int(n_max))
            r3 = float(meta["generation_ratio"])
            winner = int(meta["winner"])
            tail = float(max(1.0 - r3, 0.0))

            mu_pred = W2(float(D), float(eta)) / W2_ref
            chi2 = float(((mu_pred - mu_obs) / sigma_obs) ** 2)

            r3_list.append(r3)
            winner_list.append(winner)
            tail_list.append(tail)
            chi2_list.append(chi2)

    r3 = np.asarray(r3_list, dtype=float)
    winners = np.asarray(winner_list, dtype=int)
    tails = np.asarray(tail_list, dtype=float)
    chi2 = np.asarray(chi2_list, dtype=float)

    return {
        "case": case.name,
        "g_mode": case.g_mode,
        "n_max": int(n_max),
        "f_R3_gt_0p90": float(np.mean(r3 >= 0.90)),
        "f_R3_gt_0p95": float(np.mean(r3 >= 0.95)),
        "f_hmumu_chi2_le_4": float(np.mean(chi2 <= 4.0)),
        "best_chi2": float(np.min(chi2)),
        "f_winner_gt_3": float(np.mean(winners > 3)),
        "mean_tail_prob": float(np.mean(tails)),
        "max_tail_prob": float(np.max(tails)),
    }


def add_convergence_deltas(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    out = [dict(r) for r in rows]
    by_case: Dict[str, List[Dict[str, float]]] = {}
    for r in out:
        by_case.setdefault(str(r["case"]), []).append(r)

    for _case, arr in by_case.items():
        arr.sort(key=lambda r: int(r["n_max"]))
        ref = arr[-1]  # largest n_max as reference
        for r in arr:
            r["ref_n_max"] = int(ref["n_max"])
            r["delta_f_R3_gt_0p90_vs_ref"] = float(r["f_R3_gt_0p90"] - ref["f_R3_gt_0p90"])
            r["delta_f_hmumu_chi2_le_4_vs_ref"] = float(
                r["f_hmumu_chi2_le_4"] - ref["f_hmumu_chi2_le_4"]
            )
            r["delta_f_winner_gt_3_vs_ref"] = float(r["f_winner_gt_3"] - ref["f_winner_gt_3"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="N_max convergence scan for g_N migration impact.")
    ap.add_argument(
        "--nmax-list",
        default="20,30,40",
        help="Comma-separated N_max values.",
    )
    args = ap.parse_args()

    nmax_list = [int(x.strip()) for x in args.nmax_list.split(",") if x.strip()]
    if not nmax_list:
        raise ValueError("nmax-list is empty.")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        Case(name="baseline_fp_2d_full", g_mode="fp_2d_full"),
        Case(name="first_principles_fp_2d", g_mode="fp_2d"),
    ]

    rows: List[Dict[str, float]] = []
    for case in cases:
        for nmax in nmax_list:
            row = eval_case_nmax(case, nmax)
            rows.append(row)
            print(
                f"[run] case={case.name}, n_max={nmax}, "
                f"f_R3_gt_0p90={row['f_R3_gt_0p90']:.6f}, "
                f"f_winner_gt_3={row['f_winner_gt_3']:.6f}"
            )

    rows = add_convergence_deltas(rows)
    rows.sort(key=lambda r: (str(r["case"]), int(r["n_max"])))

    out_csv = OUTDIR / "gn_nmax_convergence.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    paper_csv = PAPER_DIR / out_csv.name
    paper_csv.write_text(out_csv.read_text())

    print(f"[saved] {out_csv}")
    print(f"[saved] {paper_csv}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
