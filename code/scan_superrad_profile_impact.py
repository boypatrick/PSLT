#!/usr/bin/env python3
"""
Quantify global-scan impact of first-principles superradiant prefactor profiles.

Profiles are extracted by extract_superrad_prefactor_1d.py and define
D-dependent channel normalizations A1(D), A2(D) relative to D_ref=12.

Cases:
  1) baseline: A1=A2=1
  2) profile-scaled: A1=A1_profile(D), A2=A2_profile(D)
  3) profile-scaled x0.5
  4) profile-scaled x2.0

Outputs:
  - output/superrad_fp_1d/superrad_profile_impact.csv
  - output/superrad_fp_1d/superrad_profile_values.csv
  - paper/superrad_profile_impact.csv
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((ROOT / "code").resolve()))

from pslt_lib import PSLTKinetics, PSLTParameters


OUTDIR = ROOT / "output" / "superrad_fp_1d"
PAPER_DIR = ROOT / "paper"


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    scale: float = 1.0


def make_kinetics() -> PSLTKinetics:
    params = PSLTParameters(
        c_eff=0.5,
        nu=5.0,
        kappa_g=0.03,
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


def load_profiles() -> Tuple[Callable[[float], float], Callable[[float], float], np.ndarray, np.ndarray, np.ndarray]:
    path = OUTDIR / "superrad_prefactor_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing superradiant profile csv: {path}")

    df = pd.read_csv(path)
    df = df[df["level"].str.lower() == "fine"].copy().sort_values("D")
    d = np.asarray(df["D"], dtype=float)
    a1 = np.asarray(df["A1_profile"], dtype=float)
    a2 = np.asarray(df["A2_profile"], dtype=float)

    def a1_fn(D: float) -> float:
        return float(np.interp(D, d, a1))

    def a2_fn(D: float) -> float:
        return float(np.interp(D, d, a2))

    return a1_fn, a2_fn, d, a1, a2


def calculate_gamma_with_profiles(
    kin: PSLTKinetics,
    N: int,
    D: float,
    eta_eff: float,
    A1_eff: float,
    A2_eff: float,
) -> float:
    if D <= 0:
        return 0.0

    p = kin.params
    mu = p.M / D
    OmegaH = p.Omega_H * p.M
    alpha = 1.0 / D

    w = kin.omega_N(mu, N, D)

    def gamma_sr(ell: int, m: int, Aeff: float) -> float:
        delta_tilde = (m * OmegaH - w) / p.M
        if delta_tilde <= 0.0:
            return 0.0
        return p.M * Aeff * (alpha ** (4 * ell + 4)) * delta_tilde

    g1 = gamma_sr(1, 1, A1_eff)
    g2 = gamma_sr(2, 2, A2_eff)

    S = kin.action_S(mu, D, w)
    r_N = eta_eff * np.exp(-2.0 * S)

    chi_eff = kin.chi_effective(D)
    eps_mix = chi_eff * math.sqrt(g1 * g2) if (g1 > 0 and g2 > 0) else 0.0

    tr = g1 + g2
    disc = (g1 - g2) ** 2 + 4.0 * (eps_mix ** 2)
    lam_plus = 0.5 * (tr + math.sqrt(disc))

    return max(float(r_N * lam_plus), 0.0)


def evaluate_case(
    case: Case,
    kin: PSLTKinetics,
    a1_fn: Callable[[float], float],
    a2_fn: Callable[[float], float],
) -> Dict[str, float]:
    D_vals = np.linspace(4.0, 20.0, 60)
    eta_vals = np.linspace(0.2, 4.0, 60)
    t_coh = 1.0
    mu_obs = 1.4
    sigma_obs = 0.4
    D0, eta0 = 10.0, 1.0

    def Aeff(D: float) -> Tuple[float, float]:
        if case.mode == "baseline":
            return 1.0, 1.0
        a1 = case.scale * a1_fn(D)
        a2 = case.scale * a2_fn(D)
        return float(a1), float(a2)

    def gamma_N(N: int, D: float, eta: float) -> float:
        A1_eff, A2_eff = Aeff(D)
        return calculate_gamma_with_profiles(kin, N=N, D=D, eta_eff=eta, A1_eff=A1_eff, A2_eff=A2_eff)

    def W2(D: float, eta: float) -> float:
        N = 2
        Gam = gamma_N(N, D, eta)
        g = kin.g_N_cardy(N)
        B = kin.B_N(N)
        return float(B * g * (1.0 - np.exp(-Gam * t_coh)))

    W2_ref = W2(D0, eta0)

    ratios = []
    chi2_vals = []
    winners = []

    for eta in eta_vals:
        for D in D_vals:
            weights = []
            gammas = []
            for N in range(1, 21):
                Gam = gamma_N(N, D, eta)
                g = kin.g_N_cardy(N)
                B = kin.B_N(N)
                wN = B * g * (1.0 - np.exp(-Gam * t_coh))
                weights.append(wN)
                gammas.append(Gam)
            weights = np.asarray(weights, dtype=float)
            total = float(np.sum(weights))
            P = weights / total if total > 0 else weights

            ratios.append(float(np.sum(P[:3])))
            winners.append(int(np.argmax(P) + 1))

            mu_pred = W2(D, eta) / W2_ref if W2_ref > 0 else 0.0
            chi2_vals.append(float(((mu_pred - mu_obs) / sigma_obs) ** 2))

    ratios = np.asarray(ratios)
    chi2_vals = np.asarray(chi2_vals)
    winners = np.asarray(winners)

    return {
        "case": case.name,
        "f_R3_gt_0p90": float(np.mean(ratios >= 0.90)),
        "f_R3_gt_0p95": float(np.mean(ratios >= 0.95)),
        "f_hmumu_chi2_le_4": float(np.mean(chi2_vals <= 4.0)),
        "best_chi2": float(np.min(chi2_vals)),
        "winner_mean": float(np.mean(winners)),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    kin = make_kinetics()
    a1_fn, a2_fn, d, a1v, a2v = load_profiles()

    cases: List[Case] = [
        Case(name="baseline_A1_A2_const", mode="baseline", scale=1.0),
        Case(name="profile_scaled_A1A2", mode="profile", scale=1.0),
        Case(name="profile_scaled_A1A2_x0p5", mode="profile", scale=0.5),
        Case(name="profile_scaled_A1A2_x2", mode="profile", scale=2.0),
    ]

    rows = [evaluate_case(c, kin, a1_fn, a2_fn) for c in cases]

    out_csv = OUTDIR / "superrad_profile_impact.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "f_R3_gt_0p90", "f_R3_gt_0p95", "f_hmumu_chi2_le_4", "best_chi2", "winner_mean"],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_profile = OUTDIR / "superrad_profile_values.csv"
    pd.DataFrame({"D": d, "A1_profile": a1v, "A2_profile": a2v}).to_csv(out_profile, index=False)

    paper_csv = PAPER_DIR / "superrad_profile_impact.csv"
    paper_csv.write_text(out_csv.read_text())

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_profile}")
    print(f"[saved] {paper_csv}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
