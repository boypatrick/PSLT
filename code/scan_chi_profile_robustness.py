#!/usr/bin/env python3
"""
Robustness scan for global metrics under different chi(D) profile choices.

Outputs:
  - output/chi_fp_2d/chi_profile_robustness.csv
  - output/chi_fp_2d/chi_profile_robustness_profiles.csv
  - paper/chi_profile_robustness.csv
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pslt_lib import PSLTKinetics, PSLTParameters


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_fp_2d"
PAPER_DIR = ROOT / "paper"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"


@dataclass(frozen=True)
class ProfileCase:
    name: str
    model: str  # linear_interp | exp_fit
    scale: float


BASELINE = {
    "c_eff": 0.5,
    "nu": 5.0,
    "kappa_g": 0.03,
    "g_mode": "fp_2d_full",
    "A1": 1.0,
    "A2": 1.0,
    "p_B": 0.30,
    "t_coh": 1.0,
    "hmumu_ref_D": 10.0,
    "hmumu_ref_eta": 1.0,
    "mu_obs": 1.4,
    "sigma_obs": 0.4,
}


def load_fine_knots() -> Tuple[np.ndarray, np.ndarray]:
    path = OUTDIR / "localized_chi_D6-12-18.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing localized extraction file: {path}")

    rows: List[Tuple[float, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("level", "").strip().lower() == "fine":
                rows.append((float(row["D"]), float(row["chi_LR"])))

    if len(rows) < 3:
        raise RuntimeError("Need at least 3 fine-grid points from localized extraction.")

    rows.sort(key=lambda t: t[0])
    d = np.array([x for x, _ in rows], dtype=float)
    chi = np.array([y for _, y in rows], dtype=float)
    return d, chi


def build_profile(case: ProfileCase, d_knots: np.ndarray, chi_knots: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d_dense = np.linspace(4.0, 20.0, 17)

    if case.model == "linear_interp":
        chi_dense = np.interp(d_dense, d_knots, chi_knots)
    elif case.model == "exp_fit":
        coeff = np.polyfit(d_knots, np.log(chi_knots), 1)  # ln chi = m D + b
        m, b = float(coeff[0]), float(coeff[1])
        chi_dense = np.exp(m * d_dense + b)
    else:
        raise ValueError(f"Unknown profile model: {case.model}")

    chi_dense = np.maximum(chi_dense * case.scale, 1e-16)
    return d_dense, chi_dense


def make_kinetics(d_profile: np.ndarray, chi_profile: np.ndarray) -> PSLTKinetics:
    params = PSLTParameters(
        c_eff=BASELINE["c_eff"],
        nu=BASELINE["nu"],
        kappa_g=BASELINE["kappa_g"],
        g_mode=BASELINE["g_mode"],
        g_fp_full_window_blend=0.8,
        g_fp_full_tail_beta=1.1,
        g_fp_full_tail_shell_power=0.0,
        g_fp_full_tail_clip_min=1e-3,
        g_fp_full_tail_clip_max=0.95,
        chi=0.2,
        chi_mode="localized_interp",
        chi_lr_D=tuple(float(x) for x in d_profile),
        chi_lr_vals=tuple(float(y) for y in chi_profile),
        A1=BASELINE["A1"],
        A2=BASELINE["A2"],
        b_mode="overlap_2d",
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=BASELINE["p_B"],
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
    )
    return PSLTKinetics(params)


def evaluate_case(case: ProfileCase, d_knots: np.ndarray, chi_knots: np.ndarray) -> Dict[str, float]:
    d_profile, chi_profile = build_profile(case, d_knots, chi_knots)
    kinetics = make_kinetics(d_profile, chi_profile)

    d_vals = np.linspace(4.0, 20.0, 60)
    eta_vals = np.linspace(0.2, 4.0, 60)
    t_coh = BASELINE["t_coh"]
    n_max = 20

    # R3 metrics.
    r3_map = np.zeros((len(eta_vals), len(d_vals)))
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            _, _, meta = kinetics.get_probabilities(d, eta, t_coh, n_max)
            r3_map[i, j] = meta["generation_ratio"]

    frac_90 = float(np.mean(r3_map >= 0.90))
    frac_95 = float(np.mean(r3_map >= 0.95))

    # H->mumu proxy metrics.
    def get_w2(d: float, eta: float) -> float:
        n = 2
        gamma = kinetics.calculate_gamma_N(n, d, eta)
        g_n = kinetics.g_N_effective(n, d)
        b_n = kinetics.B_N(n, d)
        return b_n * g_n * (1.0 - np.exp(-gamma * t_coh))

    w2_ref = get_w2(BASELINE["hmumu_ref_D"], BASELINE["hmumu_ref_eta"])
    if w2_ref <= 0:
        raise RuntimeError("Reference W2 is non-positive.")

    chi2 = np.zeros_like(r3_map)
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            mu_pred = get_w2(d, eta) / w2_ref
            chi2[i, j] = ((mu_pred - BASELINE["mu_obs"]) / BASELINE["sigma_obs"]) ** 2

    frac_hmumu = float(np.mean(chi2 <= 4.0))
    idx = np.unravel_index(np.argmin(chi2), chi2.shape)
    best_chi2 = float(chi2[idx])
    best_d = float(d_vals[idx[1]])
    best_eta = float(eta_vals[idx[0]])

    return {
        "profile": case.name,
        "model": case.model,
        "scale": case.scale,
        "f_r3_ge_90": frac_90,
        "f_r3_ge_95": frac_95,
        "f_hmumu_chi2_le_4": frac_hmumu,
        "best_chi2": best_chi2,
        "best_D": best_d,
        "best_eta": best_eta,
        "chi_at_D6": float(np.interp(6.0, d_profile, chi_profile)),
        "chi_at_D12": float(np.interp(12.0, d_profile, chi_profile)),
        "chi_at_D18": float(np.interp(18.0, d_profile, chi_profile)),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    d_knots, chi_knots = load_fine_knots()

    cases = [
        ProfileCase("linear_interp", "linear_interp", 1.0),
        ProfileCase("exp_fit", "exp_fit", 1.0),
        ProfileCase("linear_interp_x0.8", "linear_interp", 0.8),
        ProfileCase("linear_interp_x1.2", "linear_interp", 1.2),
        ProfileCase("exp_fit_x0.8", "exp_fit", 0.8),
        ProfileCase("exp_fit_x1.2", "exp_fit", 1.2),
    ]

    rows = [evaluate_case(c, d_knots, chi_knots) for c in cases]

    out_csv = OUTDIR / "chi_profile_robustness.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Export profile samples for plotting or auditing.
    profile_rows = []
    for c in cases:
        d_profile, chi_profile = build_profile(c, d_knots, chi_knots)
        for d, chi in zip(d_profile, chi_profile):
            profile_rows.append(
                {
                    "profile": c.name,
                    "D": float(d),
                    "chi": float(chi),
                }
            )
    out_profiles = OUTDIR / "chi_profile_robustness_profiles.csv"
    with open(out_profiles, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["profile", "D", "chi"])
        writer.writeheader()
        writer.writerows(profile_rows)

    # Mirror the summary csv into paper folder for direct reference in LaTeX.
    paper_csv = PAPER_DIR / "chi_profile_robustness.csv"
    with open(paper_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] wrote summary: {out_csv}")
    print(f"[done] wrote profile samples: {out_profiles}")
    print(f"[done] wrote paper copy: {paper_csv}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
