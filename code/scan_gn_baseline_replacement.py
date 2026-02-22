#!/usr/bin/env python3
"""
Baseline-replacement audit for g_N: Cardy surrogate vs fp_2d_full.

Goal:
  Promote a first-principles candidate from comparator-only to baseline-replacement
  stress test at map level, including N_max convergence checks.

Outputs:
  - output/gn_fp_impact/gn_baseline_replacement.csv
  - output/gn_fp_impact/gn_baseline_replacement_nmax.csv
  - output/gn_fp_impact/gn_baseline_replacement_profile_values.csv
  - paper/gn_baseline_replacement.csv
  - paper/gn_baseline_replacement_nmax.csv
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from pslt_lib import PSLTKinetics, PSLTParameters


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "gn_fp_impact"
PAPER_DIR = ROOT / "paper"


@dataclass(frozen=True)
class Case:
    name: str
    g_mode: str
    g_fp_blend: float


def make_kinetics(case: Case) -> PSLTKinetics:
    params = PSLTParameters(
        c_eff=0.5,
        nu=5.0,
        kappa_g=0.03,
        g_mode=case.g_mode,
        g_fp_blend=case.g_fp_blend,
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
        b_n_power=0.30,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
    )
    return PSLTKinetics(params)


def evaluate_case_nmax(case: Case, n_max: int) -> Dict[str, float]:
    kin = make_kinetics(case)

    d_vals = np.linspace(4.0, 20.0, 60)
    eta_vals = np.linspace(0.2, 4.0, 60)
    t_coh = 1.0
    mu_obs = 1.4
    sigma_obs = 0.4
    d_ref, eta_ref = 10.0, 1.0

    def w2(d: float, eta: float) -> float:
        n = 2
        gamma = kin.calculate_gamma_N(n, d, eta)
        g = kin.g_N_effective(n, d)
        b = kin.B_N(n)
        return float(b * g * (1.0 - np.exp(-gamma * t_coh)))

    w2_ref = w2(d_ref, eta_ref)
    if w2_ref <= 0:
        raise RuntimeError(f"{case.name}: non-positive W2_ref.")

    r3_vals: List[float] = []
    tail_vals: List[float] = []
    winner_vals: List[int] = []
    chi2_vals: List[float] = []

    for eta in eta_vals:
        for d in d_vals:
            _, _, meta = kin.get_probabilities(d, eta, t_coh, N_max=n_max)
            r3 = float(meta["generation_ratio"])
            tail = float(max(1.0 - r3, 0.0))
            winner = int(meta["winner"])
            mu_pred = w2(d, eta) / w2_ref
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
        "g_fp_blend": case.g_fp_blend,
        "n_max": int(n_max),
        "f_R3_gt_0p90": float(np.mean(r3_arr >= 0.90)),
        "f_R3_gt_0p95": float(np.mean(r3_arr >= 0.95)),
        "f_hmumu_chi2_le_4": float(np.mean(chi2_arr <= 4.0)),
        "best_chi2": float(np.min(chi2_arr)),
        "f_winner_gt_3": float(np.mean(win_arr > 3)),
        "mean_tail_prob": float(np.mean(tail_arr)),
        "max_tail_prob": float(np.max(tail_arr)),
        "g1_D12": float(kin.g_N_effective(1, 12.0)),
        "g2_D12": float(kin.g_N_effective(2, 12.0)),
        "g3_D12": float(kin.g_N_effective(3, 12.0)),
        "g4_D12": float(kin.g_N_effective(4, 12.0)),
        "g6_D12": float(kin.g_N_effective(6, 12.0)),
    }


def export_profile_rows(cases: List[Case]) -> List[Dict[str, float]]:
    d_grid = np.linspace(4.0, 20.0, 17)
    rows: List[Dict[str, float]] = []
    for case in cases:
        kin = make_kinetics(case)
        for d in d_grid:
            rec: Dict[str, float] = {
                "case": case.name,
                "g_mode_requested": case.g_mode,
                "g_mode_active": kin.active_g_mode(),
                "g_fp_blend": case.g_fp_blend,
                "D": float(d),
            }
            for n in range(1, 11):
                rec[f"g{n}"] = float(kin.g_N_effective(n, d))
            rows.append(rec)
    return rows


def with_deltas(rows: List[Dict[str, float]], baseline_case: str) -> List[Dict[str, float]]:
    base_row = None
    for r in rows:
        if str(r["case"]) == baseline_case:
            base_row = r
            break
    if base_row is None:
        raise RuntimeError(f"Baseline case not found: {baseline_case}")

    out: List[Dict[str, float]] = []
    for r in rows:
        row = dict(r)
        row["delta_f_R3_gt_0p90_vs_baseline"] = float(r["f_R3_gt_0p90"] - base_row["f_R3_gt_0p90"])
        row["delta_f_hmumu_chi2_le_4_vs_baseline"] = float(r["f_hmumu_chi2_le_4"] - base_row["f_hmumu_chi2_le_4"])
        row["delta_f_winner_gt_3_vs_baseline"] = float(r["f_winner_gt_3"] - base_row["f_winner_gt_3"])
        out.append(row)
    return out


def add_nmax_ref_deltas(rows: List[Dict[str, float]], ref_n_max: int) -> List[Dict[str, float]]:
    ref_by_case: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if int(r["n_max"]) == int(ref_n_max):
            ref_by_case[str(r["case"])] = r

    out: List[Dict[str, float]] = []
    for r in rows:
        ref = ref_by_case[str(r["case"])]
        row = dict(r)
        row["ref_n_max"] = int(ref_n_max)
        row["delta_f_R3_gt_0p90_vs_ref"] = float(r["f_R3_gt_0p90"] - ref["f_R3_gt_0p90"])
        row["delta_f_hmumu_chi2_le_4_vs_ref"] = float(r["f_hmumu_chi2_le_4"] - ref["f_hmumu_chi2_le_4"])
        row["delta_f_winner_gt_3_vs_ref"] = float(r["f_winner_gt_3"] - ref["f_winner_gt_3"])
        out.append(row)
    return out


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
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        Case(name="baseline_cardy", g_mode="cardy", g_fp_blend=0.01),
        Case(name="replacement_fp_2d_full", g_mode="fp_2d_full", g_fp_blend=1.0),
    ]

    nmax_list = [20, 30, 40]

    nmax_rows: List[Dict[str, float]] = []
    for case in cases:
        for n_max in nmax_list:
            nmax_rows.append(evaluate_case_nmax(case, n_max))
    nmax_rows = add_nmax_ref_deltas(nmax_rows, ref_n_max=40)

    # Summary table uses manuscript baseline N_max=20.
    summary_rows = [r for r in nmax_rows if int(r["n_max"]) == 20]
    summary_rows = with_deltas(summary_rows, baseline_case="baseline_cardy")

    profile_rows = export_profile_rows(cases)

    out_summary = OUTDIR / "gn_baseline_replacement.csv"
    out_nmax = OUTDIR / "gn_baseline_replacement_nmax.csv"
    out_profile = OUTDIR / "gn_baseline_replacement_profile_values.csv"
    paper_summary = PAPER_DIR / "gn_baseline_replacement.csv"
    paper_nmax = PAPER_DIR / "gn_baseline_replacement_nmax.csv"

    write_csv(out_summary, summary_rows)
    write_csv(out_nmax, nmax_rows)
    write_csv(out_profile, profile_rows)
    write_csv(paper_summary, summary_rows)
    write_csv(paper_nmax, nmax_rows)

    print(f"[saved] {out_summary}")
    print(f"[saved] {out_nmax}")
    print(f"[saved] {out_profile}")
    print(f"[saved] {paper_summary}")
    print(f"[saved] {paper_nmax}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
