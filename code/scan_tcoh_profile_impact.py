#!/usr/bin/env python3
"""
Quantify global-scan impact of a first-principles dephasing t_coh profile.

Cases:
  1) constant baseline: t_coh = 1
  2) dephasing profile: t_coh(D) = pi / Delta_omega_12(D)
  3) dephasing profile with cap: min(t_coh(D), t_cap)

Outputs:
  - output/tcoh_fp_1d/tcoh_profile_impact.csv
  - paper/tcoh_profile_impact.csv
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((ROOT / "code").resolve()))

from pslt_lib import PSLTKinetics, PSLTParameters
from action_grid_profile_utils import scan_d_values, select_chi_profile


OUTDIR = ROOT / "output" / "tcoh_fp_1d"
PAPER_DIR = ROOT / "paper"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"
_CHI_PROFILE: dict | None = None


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    t_cap: float | None = None


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
        b_mode="overlap_2d",
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=0.30,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode="eft_wilson_matched",
        hll_observable_nmax=20,
    )
    return PSLTKinetics(params)


def load_tcoh_profile() -> Callable[[float], float]:
    path = OUTDIR / "tcoh_dephasing_D4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dephasing profile csv: {path}")
    df = pd.read_csv(path)
    df = df[df["level"].str.lower() == "fine"].copy()
    d = np.asarray(df["D"], dtype=float)
    t = np.asarray(df["t_coh_deph"], dtype=float)
    order = np.argsort(d)
    d = d[order]
    t = t[order]

    def interp_fn(D: float) -> float:
        return float(np.interp(D, d, t))

    return interp_fn


def evaluate_case(case: Case, kin: PSLTKinetics, tcoh_fn: Callable[[float], float]) -> Dict[str, float]:
    D_vals = np.linspace(4.0, 20.0, 60)
    eta_vals = np.linspace(0.2, 4.0, 60)
    mu_obs = 1.4
    sigma_obs = 0.4
    D0, eta0 = 10.0, 1.0
    hll_mode = "eft_wilson_matched"
    hll_nmax = 20

    def t_coh_of_D(D: float) -> float:
        if case.mode == "constant":
            return 1.0
        val = tcoh_fn(D)
        if case.mode == "dephasing_capped":
            assert case.t_cap is not None
            val = min(val, case.t_cap)
        return float(val)

    ref_t = t_coh_of_D(D0)
    amp_ref = kin.hll_channel_amplitude(2, D0, eta0, ref_t, observable_mode=hll_mode, N_max=hll_nmax)
    if amp_ref <= 0:
        raise RuntimeError(f"{case.name}: non-positive H->mumu reference amplitude.")

    def mu_mumu(D: float, eta: float) -> float:
        amp = kin.hll_channel_amplitude(
            2,
            D,
            eta,
            t_coh_of_D(D),
            observable_mode=hll_mode,
            N_max=hll_nmax,
        )
        ratio = float(amp / max(amp_ref, 1e-30))
        if hll_mode == "proxy_wratio":
            return ratio
        return ratio * ratio

    ratios = []
    chi2_vals = []
    for eta in eta_vals:
        for D in D_vals:
            _, _, meta = kin.get_probabilities(D, eta, t_coh_of_D(D), N_max=20)
            ratios.append(float(meta["generation_ratio"]))
            mu_pred = mu_mumu(D, eta)
            chi2_vals.append(float(((mu_pred - mu_obs) / sigma_obs) ** 2))

    ratios = np.asarray(ratios)
    chi2_vals = np.asarray(chi2_vals)
    return {
        "case": case.name,
        "f_R3_gt_0p90": float(np.mean(ratios >= 0.90)),
        "f_R3_gt_0p95": float(np.mean(ratios >= 0.95)),
        "f_hmumu_chi2_le_4": float(np.mean(chi2_vals <= 4.0)),
        "best_chi2": float(np.min(chi2_vals)),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    kin = make_kinetics()
    tcoh_fn = load_tcoh_profile()

    cases: List[Case] = [
        Case(name="constant_tcoh_1", mode="constant"),
        Case(name="dephasing_pi_over_deltaomega", mode="dephasing"),
        Case(name="dephasing_capped_1e4", mode="dephasing_capped", t_cap=1.0e4),
    ]

    rows = [evaluate_case(c, kin, tcoh_fn) for c in cases]
    out_csv = OUTDIR / "tcoh_profile_impact.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "f_R3_gt_0p90", "f_R3_gt_0p95", "f_hmumu_chi2_le_4", "best_chi2"],
        )
        writer.writeheader()
        writer.writerows(rows)

    paper_csv = PAPER_DIR / "tcoh_profile_impact.csv"
    paper_csv.write_text(out_csv.read_text())

    print(f"[saved] {out_csv}")
    print(f"[saved] {paper_csv}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
