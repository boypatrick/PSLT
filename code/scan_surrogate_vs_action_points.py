#!/usr/bin/env python3
"""
Point-level bias audit: surrogate interpolation vs direct action extraction.

B3 objective:
  Quantify transfer bias from the fast surrogate chain to a direct
  action-derived localized extraction at representative (D, eta) points.

Comparison setup:
  - Surrogate chain: chi_mode="localized_interp" using D={6,12,18} fine knots.
  - Direct chain: chi_mode="constant", where chi is replaced by direct
    localized fine extraction at each queried D (slow path).

Outputs:
  - output/chi_fp_2d/surrogate_vs_action_points.csv
  - output/chi_fp_2d/surrogate_vs_action_points_summary.csv
  - paper/surrogate_vs_action_points.csv
  - paper/surrogate_vs_action_points_summary.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from extract_chi_localized_2d import Level, PhysicalParams, run_case
from pslt_lib import PSLTKinetics, PSLTParameters


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_fp_2d"
PAPER_DIR = ROOT / "paper"


BASELINE = {
    "c_eff": 0.5,
    "nu": 5.0,
    "kappa_g": 0.03,
    "A1": 1.0,
    "A2": 1.0,
    "p_B": 0.30,
    "t_coh": 1.0,
    "n_max": 20,
    "hmumu_ref_D": 10.0,
    "hmumu_ref_eta": 1.0,
}


POINTS = [
    {"name": "gen3_zone", "D": 4.0, "eta": 0.2},
    {"name": "gen2_zone", "D": 5.0, "eta": 2.0},
    {"name": "gen1_zone", "D": 11.0, "eta": 1.4},
]


def load_surrogate_knots() -> Tuple[np.ndarray, np.ndarray]:
    path = OUTDIR / "localized_chi_D6-12-18.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing chi knot file: {path}")
    rows: List[Tuple[float, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("level", "").strip().lower() == "fine":
                rows.append((float(row["D"]), float(row["chi_LR"])))
    if len(rows) < 3:
        raise RuntimeError(f"Not enough fine rows in {path}")
    rows.sort(key=lambda t: t[0])
    d = np.array([x for x, _ in rows], dtype=float)
    chi = np.array([y for _, y in rows], dtype=float)
    return d, chi


def make_kinetics_surrogate(d_knots: np.ndarray, chi_knots: np.ndarray) -> PSLTKinetics:
    params = PSLTParameters(
        c_eff=BASELINE["c_eff"],
        nu=BASELINE["nu"],
        kappa_g=BASELINE["kappa_g"],
        chi=0.2,
        chi_mode="localized_interp",
        chi_lr_D=tuple(float(x) for x in d_knots),
        chi_lr_vals=tuple(float(y) for y in chi_knots),
        A1=BASELINE["A1"],
        A2=BASELINE["A2"],
        b_n_power=BASELINE["p_B"],
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
    )
    return PSLTKinetics(params)


def make_kinetics_constant(chi_const: float) -> PSLTKinetics:
    params = PSLTParameters(
        c_eff=BASELINE["c_eff"],
        nu=BASELINE["nu"],
        kappa_g=BASELINE["kappa_g"],
        chi=float(chi_const),
        chi_mode="constant",
        A1=BASELINE["A1"],
        A2=BASELINE["A2"],
        b_n_power=BASELINE["p_B"],
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
    )
    return PSLTKinetics(params)


def compute_w2(kin: PSLTKinetics, d: float, eta: float) -> float:
    n = 2
    gamma = kin.calculate_gamma_N(n, d, eta)
    g_n = kin.g_N_effective(n, d)
    b_n = kin.B_N(n)
    return float(b_n * g_n * (1.0 - np.exp(-gamma * BASELINE["t_coh"])))


def extract_direct_chi(d_values: List[float]) -> Dict[float, Dict[str, float]]:
    p = PhysicalParams()
    fine = Level("fine", dr=0.06, dz=0.03)
    out: Dict[float, Dict[str, float]] = {}
    for d in sorted(set(float(x) for x in d_values)):
        row = run_case(
            D=d,
            level=fine,
            p=p,
            rho_max=3.0,
            z_margin=6.0,
            n_mu=120,
            tol=1e-8,
            maxiter=30000,
            sigma=2.5,
        )
        out[d] = {
            "chi_LR": float(row["chi_LR"]),
            "solve_s": float(row["solve_s"]),
            "build_s": float(row["build_s"]),
        }
    return out


def point_metrics(kin: PSLTKinetics, d: float, eta: float) -> Dict[str, float]:
    _, p_n, meta = kin.get_probabilities(d, eta, BASELINE["t_coh"], N_max=BASELINE["n_max"])
    return {
        "winner": float(meta["winner"]),
        "R3": float(meta["generation_ratio"]),
        "P1": float(p_n[0]) if len(p_n) >= 1 else 0.0,
        "P2": float(p_n[1]) if len(p_n) >= 2 else 0.0,
        "P3": float(p_n[2]) if len(p_n) >= 3 else 0.0,
        "W2": compute_w2(kin, d, eta),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    d_knots, chi_knots = load_surrogate_knots()
    kin_sur = make_kinetics_surrogate(d_knots, chi_knots)

    ref_d = float(BASELINE["hmumu_ref_D"])
    ref_eta = float(BASELINE["hmumu_ref_eta"])
    chi_ref_direct_targets = [ref_d]
    point_ds = [float(p["D"]) for p in POINTS]
    direct = extract_direct_chi(point_ds + chi_ref_direct_targets)

    w2_ref_sur = compute_w2(kin_sur, ref_d, ref_eta)
    kin_ref_dir = make_kinetics_constant(direct[ref_d]["chi_LR"])
    w2_ref_dir = compute_w2(kin_ref_dir, ref_d, ref_eta)

    rows = []
    for p in POINTS:
        name = str(p["name"])
        d = float(p["D"])
        eta = float(p["eta"])

        chi_interp = float(kin_sur.chi_effective(d))
        chi_direct = float(direct[d]["chi_LR"])
        kin_dir = make_kinetics_constant(chi_direct)

        m_sur = point_metrics(kin_sur, d, eta)
        m_dir = point_metrics(kin_dir, d, eta)

        mu_sur = m_sur["W2"] / max(w2_ref_sur, 1e-30)
        mu_dir = m_dir["W2"] / max(w2_ref_dir, 1e-30)

        delta_mu_abs = abs(mu_dir - mu_sur)
        delta_mu_rel = (mu_dir - mu_sur) / max(abs(mu_sur), 1e-30)
        delta_chi_rel = (chi_direct - chi_interp) / max(abs(chi_interp), 1e-30)

        rows.append(
            {
                "point": name,
                "D": d,
                "eta": eta,
                "chi_interp": chi_interp,
                "chi_direct": chi_direct,
                "delta_chi_rel": delta_chi_rel,
                "winner_surrogate": int(round(m_sur["winner"])),
                "winner_direct": int(round(m_dir["winner"])),
                "winner_match": int(int(round(m_sur["winner"])) == int(round(m_dir["winner"]))),
                "R3_surrogate": m_sur["R3"],
                "R3_direct": m_dir["R3"],
                "delta_R3_abs": abs(m_dir["R3"] - m_sur["R3"]),
                "P1_surrogate": m_sur["P1"],
                "P2_surrogate": m_sur["P2"],
                "P3_surrogate": m_sur["P3"],
                "P1_direct": m_dir["P1"],
                "P2_direct": m_dir["P2"],
                "P3_direct": m_dir["P3"],
                "mu_mumu_surrogate": mu_sur,
                "mu_mumu_direct": mu_dir,
                "delta_mu_abs": delta_mu_abs,
                "delta_mu_rel": delta_mu_rel,
                "solve_s_direct": float(direct[d]["solve_s"]),
                "build_s_direct": float(direct[d]["build_s"]),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "n_points": int(len(df)),
        "frac_winner_match": float(df["winner_match"].mean()) if len(df) else 0.0,
        "max_abs_delta_R3": float(df["delta_R3_abs"].max()) if len(df) else 0.0,
        "max_abs_delta_mu": float(df["delta_mu_abs"].max()) if len(df) else 0.0,
        "max_abs_rel_delta_chi": float(np.abs(df["delta_chi_rel"]).max()) if len(df) else 0.0,
    }
    df_sum = pd.DataFrame([summary])

    out_csv = OUTDIR / "surrogate_vs_action_points.csv"
    out_sum = OUTDIR / "surrogate_vs_action_points_summary.csv"
    df.to_csv(out_csv, index=False)
    df_sum.to_csv(out_sum, index=False)

    paper_csv = PAPER_DIR / out_csv.name
    paper_sum = PAPER_DIR / out_sum.name
    df.to_csv(paper_csv, index=False)
    df_sum.to_csv(paper_sum, index=False)

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_sum}")
    print(f"[saved] {paper_csv}")
    print(f"[saved] {paper_sum}")
    print(df.to_string(index=False))
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
