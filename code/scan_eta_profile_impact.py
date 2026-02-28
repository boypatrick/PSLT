#!/usr/bin/env python3
"""
Quantify global-scan impact of first-principles eta(D) prefactor candidates.

Cases:
  1) baseline: eta_eff = eta
  2) scaled by amplitude profile: eta_eff = eta * eta_amp(D)
  3) scaled by probability profile: eta_eff = eta * eta_prob(D)
  4) closed amplitude profile: eta_eff = eta_amp(D) (eta-axis ignored)
  5) closed probability profile: eta_eff = eta_prob(D) (eta-axis ignored)

Outputs:
  - output/eta_fp_1d/eta_profile_impact.csv
  - output/eta_fp_1d/eta_profile_values.csv
  - paper/eta_profile_impact.csv
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((ROOT / "code").resolve()))

from pslt_lib import PSLTKinetics, PSLTParameters
from action_grid_profile_utils import scan_d_values, select_chi_profile


OUTDIR = ROOT / "output" / "eta_fp_1d"
PAPER_DIR = ROOT / "paper"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"
_CHI_PROFILE: dict | None = None


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    profile: str


def get_chi_profile() -> dict:
    global _CHI_PROFILE
    if _CHI_PROFILE is None:
        d_scan = scan_d_values(4.0, 20.0, 60)
        _CHI_PROFILE = select_chi_profile(ROOT, d_scan)
    return _CHI_PROFILE


def make_kinetics() -> PSLTKinetics:
    chi_profile = get_chi_profile()
    params = PSLTParameters(
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
        chi_mode=str(chi_profile["mode"]),
        chi_lr_D=tuple(float(x) for x in chi_profile["d"]),
        chi_lr_vals=tuple(float(y) for y in chi_profile["chi"]),
        A1=1.0,
        A2=1.0,
        b_mode="eft_operator_norm",
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=0.30,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode="eft_wilson_uv_rge",
        hll_observable_nmax=20,
    )
    return PSLTKinetics(params)


def load_profiles() -> Tuple[Callable[[float], float], Callable[[float], float], np.ndarray, np.ndarray, np.ndarray]:
    path = OUTDIR / "eta_prefactor_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing eta profile csv: {path}")
    df = pd.read_csv(path)
    df = df[df["level"].str.lower() == "fine"].copy().sort_values("D")
    d = np.asarray(df["D"], dtype=float)
    amp = np.asarray(df["eta_amp"], dtype=float)
    prob = np.asarray(df["eta_prob"], dtype=float)

    def amp_fn(D: float) -> float:
        return float(np.interp(D, d, amp))

    def prob_fn(D: float) -> float:
        return float(np.interp(D, d, prob))

    return amp_fn, prob_fn, d, amp, prob


def evaluate_case(case: Case, kin: PSLTKinetics, amp_fn: Callable[[float], float], prob_fn: Callable[[float], float]) -> Dict[str, float]:
    D_vals = np.linspace(4.0, 20.0, 60)
    eta_vals = np.linspace(0.2, 4.0, 60)
    t_coh = 1.0
    mu_obs = 1.4
    sigma_obs = 0.4
    D0, eta0 = 10.0, 1.0
    hll_mode = "eft_wilson_uv_rge"
    hll_nmax = 20

    def profile_of_D(D: float) -> float:
        return amp_fn(D) if case.profile == "amp" else prob_fn(D)

    def eta_eff(D: float, eta: float) -> float:
        if case.mode == "baseline":
            return float(eta)
        if case.mode == "scaled":
            return float(eta * profile_of_D(D))
        if case.mode == "closed":
            return float(profile_of_D(D))
        raise ValueError(f"Unknown mode={case.mode}")

    def get_meta(D: float, eta: float) -> Dict[str, np.ndarray]:
        e_eff = eta_eff(D, eta)
        return kin.get_probabilities(D, e_eff, t_coh, N_max=20)[2]

    ref_eta_eff = eta_eff(D0, eta0)
    amp_ref = kin.hll_channel_amplitude(
        2,
        D0,
        ref_eta_eff,
        t_coh,
        observable_mode=hll_mode,
        N_max=hll_nmax,
    )
    if amp_ref <= 0:
        raise RuntimeError(f"{case.name}: non-positive H->mumu reference amplitude.")

    def mu_mumu(D: float, eta: float) -> float:
        e_eff = eta_eff(D, eta)
        amp = kin.hll_channel_amplitude(
            2,
            D,
            e_eff,
            t_coh,
            observable_mode=hll_mode,
            N_max=hll_nmax,
        )
        ratio = float(amp / max(amp_ref, 1e-30))
        if hll_mode == "proxy_wratio":
            return ratio
        return ratio * ratio

    r3 = []
    chi2 = []
    winners = []
    for eta in eta_vals:
        for D in D_vals:
            meta = get_meta(D, eta)
            r3.append(float(meta["generation_ratio"]))
            winners.append(int(meta["winner"]))
            mu_pred = mu_mumu(D, eta)
            chi2.append(float(((mu_pred - mu_obs) / sigma_obs) ** 2))

    r3 = np.asarray(r3)
    chi2 = np.asarray(chi2)
    winners = np.asarray(winners)

    return {
        "case": case.name,
        "f_R3_gt_0p90": float(np.mean(r3 >= 0.90)),
        "f_R3_gt_0p95": float(np.mean(r3 >= 0.95)),
        "f_hmumu_chi2_le_4": float(np.mean(chi2 <= 4.0)),
        "best_chi2": float(np.min(chi2)),
        "winner_mean": float(np.mean(winners)),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    kin = make_kinetics()
    amp_fn, prob_fn, d, amp, prob = load_profiles()

    cases: List[Case] = [
        Case(name="baseline_eta_scan", mode="baseline", profile="amp"),
        Case(name="eta_scaled_amp_profile", mode="scaled", profile="amp"),
        Case(name="eta_scaled_prob_profile", mode="scaled", profile="prob"),
        Case(name="eta_closed_amp_profile", mode="closed", profile="amp"),
        Case(name="eta_closed_prob_profile", mode="closed", profile="prob"),
    ]

    rows = [evaluate_case(c, kin, amp_fn, prob_fn) for c in cases]
    out_csv = OUTDIR / "eta_profile_impact.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "f_R3_gt_0p90", "f_R3_gt_0p95", "f_hmumu_chi2_le_4", "best_chi2", "winner_mean"],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_profile = OUTDIR / "eta_profile_values.csv"
    pd.DataFrame({"D": d, "eta_amp": amp, "eta_prob": prob}).to_csv(out_profile, index=False)

    paper_csv = PAPER_DIR / "eta_profile_impact.csv"
    paper_csv.write_text(out_csv.read_text())

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_profile}")
    print(f"[saved] {paper_csv}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
