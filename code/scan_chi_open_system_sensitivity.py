#!/usr/bin/env python3
"""
Open-system chi sensitivity scan.

This script compares baseline localized chi(D) against open-system chi(D)
constructed from profile-interpolated (delta, gamma_phi, gamma_mix, gamma_ref)
through the Lindblad proxy in pslt_lib (chi_mode="open_system").

Outputs:
  - output/chi_open_system/chi_open_system_sensitivity.csv
  - output/chi_open_system/chi_open_ratio_profiles.csv
  - output/chi_open_system/chi_open_ratio_band.csv
  - paper/chi_open_system_sensitivity.csv
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((ROOT / "code").resolve()))

from pslt_lib import PSLTParameters, PSLTKinetics


OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"
CHI_FP_DIR = ROOT / "output" / "chi_fp_2d"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"


@dataclass(frozen=True)
class Case:
    name: str
    mode: str  # localized | open_system
    phi_scale: float = 1.0
    mix_scale: float = 1.0


def load_localized_profile() -> Tuple[np.ndarray, np.ndarray]:
    cands = [
        CHI_FP_DIR / "localized_chi_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv",
        CHI_FP_DIR / "localized_chi_D6-12-18.csv",
    ]
    path = next((p for p in cands if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Missing localized chi profile CSV in output/chi_fp_2d.")

    df = pd.read_csv(path)
    df = df[df["level"].str.lower() == "fine"].copy().sort_values("D")
    d = np.asarray(df["D"], dtype=float)
    chi = np.asarray(df["chi_LR"], dtype=float)
    if len(d) < 2:
        raise RuntimeError(f"Not enough localized fine-grid points in {path}.")
    return d, chi


def pick_open_csv() -> Path:
    cands = sorted(OUTDIR.glob("chi_open_system_geometry_D*.csv"))
    if not cands:
        raise FileNotFoundError("No chi_open_system_geometry_D*.csv found in output/chi_open_system.")

    best: Optional[Path] = None
    best_n = -1
    for p in cands:
        try:
            n = len(pd.read_csv(p))
        except Exception:
            continue
        if n > best_n:
            best_n = n
            best = p
    if best is None:
        raise RuntimeError("Could not parse any chi_open_system geometry CSV.")
    return best


def make_kinetics(case: Case, d_loc: np.ndarray, chi_loc: np.ndarray, open_csv: Path) -> PSLTKinetics:
    base = dict(
        c_eff=0.5,
        nu=5.0,
        kappa_g=0.03,
        g_mode="fp_2d_full",
        g_fp_full_window_blend=0.8,
        g_fp_full_tail_beta=1.1,
        g_fp_full_tail_shell_power=0.0,
        g_fp_full_tail_clip_min=1e-3,
        g_fp_full_tail_clip_max=0.95,
        chi=0.2,
        A1=1.0,
        A2=1.0,
        b_mode="overlap_2d",
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=0.30,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        chi_lr_D=tuple(float(x) for x in d_loc),
        chi_lr_vals=tuple(float(x) for x in chi_loc),
    )
    if case.mode == "localized":
        params = PSLTParameters(chi_mode="localized_interp", **base)
    else:
        params = PSLTParameters(
            chi_mode="open_system",
            chi_open_csv=str(open_csv),
            chi_open_phi_scale=case.phi_scale,
            chi_open_mix_scale=case.mix_scale,
            chi_open_tmax=200.0,
            chi_open_nstep=300,
            **base,
        )
    return PSLTKinetics(params)


def eval_case(case: Case, d_loc: np.ndarray, chi_loc: np.ndarray, open_csv: Path) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    kin = make_kinetics(case, d_loc, chi_loc, open_csv)
    D_vals = np.linspace(4.0, 20.0, 60)
    eta_vals = np.linspace(0.2, 4.0, 60)
    t_coh = 1.0
    n_max = 20
    mu_obs = 1.4
    sigma_obs = 0.4
    D0, eta0 = 10.0, 1.0

    def chi_loc_interp(D: float) -> float:
        return float(np.interp(D, d_loc, chi_loc))

    ratio_rows: List[Dict[str, float]] = []
    for D in D_vals:
        chi_eff = float(kin.chi_effective(float(D)))
        base = max(chi_loc_interp(float(D)), 1e-30)
        ratio_rows.append(
            {
                "case": case.name,
                "mode": case.mode,
                "phi_scale": case.phi_scale,
                "mix_scale": case.mix_scale,
                "D": float(D),
                "chi_eff": chi_eff,
                "chi_lr_baseline": base,
                "ratio_to_baseline": chi_eff / base,
            }
        )

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
    chi2_list: List[float] = []
    winner_list: List[int] = []
    for eta in eta_vals:
        for D in D_vals:
            _, _, meta = kin.get_probabilities(float(D), float(eta), t_coh, N_max=n_max)
            r3_list.append(float(meta["generation_ratio"]))
            winner_list.append(int(meta["winner"]))
            mu_pred = W2(float(D), float(eta)) / W2_ref
            chi2_list.append(float(((mu_pred - mu_obs) / sigma_obs) ** 2))

    r3 = np.asarray(r3_list, dtype=float)
    chi2 = np.asarray(chi2_list, dtype=float)
    winners = np.asarray(winner_list, dtype=int)
    ratio_arr = np.asarray([r["ratio_to_baseline"] for r in ratio_rows], dtype=float)

    metrics = {
        "case": case.name,
        "mode": case.mode,
        "phi_scale": case.phi_scale,
        "mix_scale": case.mix_scale,
        "chi_mode_active": kin.active_chi_mode(),
        "f_R3_gt_0p90": float(np.mean(r3 >= 0.90)),
        "f_R3_gt_0p95": float(np.mean(r3 >= 0.95)),
        "f_hmumu_chi2_le_4": float(np.mean(chi2 <= 4.0)),
        "best_chi2": float(np.min(chi2)),
        "f_winner_gt_3": float(np.mean(winners > 3)),
        "ratio_min": float(np.min(ratio_arr)),
        "ratio_max": float(np.max(ratio_arr)),
        "ratio_mean": float(np.mean(ratio_arr)),
        "ratio_std": float(np.std(ratio_arr)),
        "ratio_p10": float(np.percentile(ratio_arr, 10.0)),
        "ratio_p90": float(np.percentile(ratio_arr, 90.0)),
    }
    return metrics, ratio_rows


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    d_loc, chi_loc = load_localized_profile()
    open_csv = pick_open_csv()

    cases = [
        Case("baseline_localized", mode="localized"),
        Case("open_system_base", mode="open_system", phi_scale=1.0, mix_scale=1.0),
        Case("open_phi_x0p5", mode="open_system", phi_scale=0.5, mix_scale=1.0),
        Case("open_phi_x2", mode="open_system", phi_scale=2.0, mix_scale=1.0),
        Case("open_mix_x0p5", mode="open_system", phi_scale=1.0, mix_scale=0.5),
        Case("open_mix_x2", mode="open_system", phi_scale=1.0, mix_scale=2.0),
        Case("open_phi_x0p5_mix_x2", mode="open_system", phi_scale=0.5, mix_scale=2.0),
        Case("open_phi_x2_mix_x0p5", mode="open_system", phi_scale=2.0, mix_scale=0.5),
    ]

    metric_rows: List[Dict[str, float]] = []
    ratio_rows_all: List[Dict[str, float]] = []
    for case in cases:
        metrics, ratio_rows = eval_case(case, d_loc, chi_loc, open_csv)
        metric_rows.append(metrics)
        ratio_rows_all.extend(ratio_rows)

    out_sens = OUTDIR / "chi_open_system_sensitivity.csv"
    with out_sens.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    out_ratio = OUTDIR / "chi_open_ratio_profiles.csv"
    with out_ratio.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ratio_rows_all[0].keys()))
        writer.writeheader()
        writer.writerows(ratio_rows_all)

    base_ratio = [r for r in ratio_rows_all if r["case"] == "open_system_base"]
    arr = np.asarray([r["ratio_to_baseline"] for r in base_ratio], dtype=float)
    out_band = OUTDIR / "chi_open_ratio_band.csv"
    pd.DataFrame(
        [
            {
                "case": "open_system_base",
                "csv_profile": str(open_csv),
                "ratio_min": float(np.min(arr)),
                "ratio_max": float(np.max(arr)),
                "ratio_mean": float(np.mean(arr)),
                "ratio_std": float(np.std(arr)),
                "ratio_p10": float(np.percentile(arr, 10.0)),
                "ratio_p90": float(np.percentile(arr, 90.0)),
            }
        ]
    ).to_csv(out_band, index=False)

    paper_csv = PAPER_DIR / "chi_open_system_sensitivity.csv"
    paper_csv.write_text(out_sens.read_text())

    print(f"[saved] {out_sens}")
    print(f"[saved] {out_ratio}")
    print(f"[saved] {out_band}")
    print(f"[saved] {paper_csv}")
    for r in metric_rows:
        print(r)


if __name__ == "__main__":
    main()
