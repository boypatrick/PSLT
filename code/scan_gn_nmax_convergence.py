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
from hll_observable import HLLObservableConfig, HLLChannelPredictor
from action_grid_profile_utils import scan_d_values, select_chi_profile, select_superrad_profile


OUTDIR = ROOT / "output" / "gn_fp_impact"
PAPER_DIR = ROOT / "paper"
G_FP_BLEND = 1.0
G_FP_FULL_WINDOW_BLEND = 0.8
G_FP_FULL_TAIL_BETA = 1.1
G_FP_FULL_TAIL_SHELL_POWER = 0.0
G_FP_FULL_TAIL_CLIP_MIN = 1e-3
G_FP_FULL_TAIL_CLIP_MAX = 0.95
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"
_ACTION_PROFILES: tuple[dict, dict] | None = None


@dataclass(frozen=True)
class Case:
    name: str
    g_mode: str
    g_fp_blend: float


def get_action_profiles() -> tuple[dict, dict]:
    global _ACTION_PROFILES
    if _ACTION_PROFILES is None:
        d_scan = scan_d_values(4.0, 20.0, 60)
        _ACTION_PROFILES = (select_chi_profile(ROOT, d_scan), select_superrad_profile(ROOT, d_scan))
    return _ACTION_PROFILES


def make_kinetics(case: Case) -> PSLTKinetics:
    chi_profile, superrad_profile = get_action_profiles()
    params = PSLTParameters(
        c_eff=0.5,
        nu=5.0,
        kappa_g=0.03,
        g_mode=case.g_mode,
        g_fp_norm_mode="phase_space",
        g_fp_blend=case.g_fp_blend,
        g_fp_full_window_blend=G_FP_FULL_WINDOW_BLEND,
        g_fp_full_tail_beta=G_FP_FULL_TAIL_BETA,
        g_fp_full_tail_shell_power=G_FP_FULL_TAIL_SHELL_POWER,
        g_fp_full_tail_clip_min=G_FP_FULL_TAIL_CLIP_MIN,
        g_fp_full_tail_clip_max=G_FP_FULL_TAIL_CLIP_MAX,
        chi=0.2,
        chi_mode=str(chi_profile["mode"]),
        chi_lr_D=tuple(float(x) for x in chi_profile["d"]),
        chi_lr_vals=tuple(float(y) for y in chi_profile["chi"]),
        A1=1.0,
        A2=1.0,
        gamma_mode=str(superrad_profile["mode"]),
        gamma_superrad_csv=str(superrad_profile["path"]),
        b_mode="eft_operator_norm",
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=0.30,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode="eft_wilson_uv_rge",
        hll_observable_nmax=20,
    )
    return PSLTKinetics(params)


def eval_case_nmax(case: Case, n_max: int) -> Dict[str, float]:
    kin = make_kinetics(case)
    D_vals = np.linspace(4.0, 20.0, 60)
    eta_vals = np.linspace(0.2, 4.0, 60)
    t_coh = 1.0
    mu_obs = 1.4
    sigma_obs = 0.4
    D0, eta0 = 10.0, 1.0
    hll_cfg = HLLObservableConfig(
        mode="eft_wilson_uv_rge",
        t_coh=t_coh,
        ref_D=D0,
        ref_eta=eta0,
        n_max=20,
    )
    hll_mumu = HLLChannelPredictor(kin, layer_n=2, cfg=hll_cfg)

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

            mu_pred = hll_mumu.mu_pred(float(D), float(eta))
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
        "g_fp_blend": float(case.g_fp_blend),
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
        Case(name="baseline_fp_2d_full", g_mode="fp_2d_full", g_fp_blend=G_FP_BLEND),
        Case(name="first_principles_fp_2d", g_mode="fp_2d", g_fp_blend=0.01),
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
