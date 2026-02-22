#!/usr/bin/env python3
"""
Map-level robustness scan for baseline closure parameters.

B2 objective:
  Quantify one-at-a-time sensitivity of key map fractions under
  reasonable finite-N surrogate ranges for (c_eff, nu, p_B).

Outputs:
  - output/robustness/core_param_robustness_cases.csv
  - output/robustness/core_param_robustness_table.csv
  - paper/core_param_robustness.csv
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pslt_lib import PSLTKinetics, PSLTParameters


ROOT = Path(__file__).resolve().parent.parent
CHI_DIR = ROOT / "output" / "chi_fp_2d"
OUTDIR = ROOT / "output" / "robustness"
PAPER_DIR = ROOT / "paper"


BASELINE = {
    "c_eff": 0.5,
    "nu": 5.0,
    "kappa_g": 0.03,
    "g_mode": "fp_2d_full",
    "g_fp_full_window_blend": 0.8,
    "g_fp_full_tail_beta": 1.1,
    "g_fp_full_tail_shell_power": 0.0,
    "g_fp_full_tail_clip_min": 1e-3,
    "g_fp_full_tail_clip_max": 0.95,
    "A1": 1.0,
    "A2": 1.0,
    "p_B": 0.30,
    "b_mode": "overlap_2d",
    "t_coh": 1.0,
    "n_max": 20,
    "D_min": 4.0,
    "D_max": 20.0,
    "D_num": 60,
    "eta_min": 0.2,
    "eta_max": 4.0,
    "eta_num": 60,
    "hmumu_ref_D": 10.0,
    "hmumu_ref_eta": 1.0,
    "mu_obs": 1.4,
    "sigma_obs": 0.4,
}
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"


@dataclass(frozen=True)
class Case:
    name: str
    c_eff: float
    nu: float
    p_B: float


def load_chi_knots() -> Tuple[np.ndarray, np.ndarray]:
    path = CHI_DIR / "localized_chi_D6-12-18.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing chi-knot file: {path}")

    rows: List[Tuple[float, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("level", "").strip().lower() == "fine":
                rows.append((float(row["D"]), float(row["chi_LR"])))

    if len(rows) < 3:
        raise RuntimeError(f"Not enough fine-grid rows in {path}")

    rows.sort(key=lambda t: t[0])
    d = np.array([x for x, _ in rows], dtype=float)
    chi = np.array([y for _, y in rows], dtype=float)
    return d, chi


def make_kinetics(case: Case, d_knots: np.ndarray, chi_knots: np.ndarray) -> PSLTKinetics:
    params = PSLTParameters(
        c_eff=case.c_eff,
        nu=case.nu,
        kappa_g=BASELINE["kappa_g"],
        g_mode=BASELINE["g_mode"],
        g_fp_full_window_blend=BASELINE["g_fp_full_window_blend"],
        g_fp_full_tail_beta=BASELINE["g_fp_full_tail_beta"],
        g_fp_full_tail_shell_power=BASELINE["g_fp_full_tail_shell_power"],
        g_fp_full_tail_clip_min=BASELINE["g_fp_full_tail_clip_min"],
        g_fp_full_tail_clip_max=BASELINE["g_fp_full_tail_clip_max"],
        chi=0.2,
        chi_mode="localized_interp",
        chi_lr_D=tuple(float(x) for x in d_knots),
        chi_lr_vals=tuple(float(y) for y in chi_knots),
        A1=BASELINE["A1"],
        A2=BASELINE["A2"],
        b_mode=BASELINE["b_mode"],
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=case.p_B,
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
    )
    return PSLTKinetics(params)


def get_w2(kin: PSLTKinetics, d: float, eta: float) -> float:
    n = 2
    gamma = kin.calculate_gamma_N(n, d, eta)
    g_n = kin.g_N_effective(n, d)
    b_n = kin.B_N(n, d)
    return float(b_n * g_n * (1.0 - np.exp(-gamma * BASELINE["t_coh"])))


def evaluate_case(case: Case, d_knots: np.ndarray, chi_knots: np.ndarray) -> Dict[str, float]:
    kin = make_kinetics(case, d_knots, chi_knots)

    d_vals = np.linspace(BASELINE["D_min"], BASELINE["D_max"], BASELINE["D_num"])
    eta_vals = np.linspace(BASELINE["eta_min"], BASELINE["eta_max"], BASELINE["eta_num"])

    r3_map = np.zeros((len(eta_vals), len(d_vals)))
    win_map = np.zeros_like(r3_map)
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            _, _, meta = kin.get_probabilities(d, eta, BASELINE["t_coh"], N_max=BASELINE["n_max"])
            r3_map[i, j] = float(meta["generation_ratio"])
            win_map[i, j] = float(meta["winner"])

    frac_r3_90 = float(np.mean(r3_map >= 0.90))
    frac_win_gt3 = float(np.mean(win_map > 3))

    w2_ref = get_w2(kin, BASELINE["hmumu_ref_D"], BASELINE["hmumu_ref_eta"])
    if w2_ref <= 0:
        raise RuntimeError(f"Non-positive W2 reference for case {case.name}: {w2_ref}")

    chi2 = np.zeros_like(r3_map)
    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            mu_pred = get_w2(kin, d, eta) / w2_ref
            chi2[i, j] = ((mu_pred - BASELINE["mu_obs"]) / BASELINE["sigma_obs"]) ** 2

    frac_chi2_le4 = float(np.mean(chi2 <= 4.0))

    return {
        "case": case.name,
        "c_eff": case.c_eff,
        "nu": case.nu,
        "p_B": case.p_B,
        "f_r3_gt_090": frac_r3_90,
        "f_chi2_le_4": frac_chi2_le4,
        "f_nwin_gt_3": frac_win_gt3,
    }


def build_cases() -> List[Case]:
    b = BASELINE
    return [
        Case("baseline", b["c_eff"], b["nu"], b["p_B"]),
        Case("c_eff_minus", 0.45, b["nu"], b["p_B"]),
        Case("c_eff_plus", 0.55, b["nu"], b["p_B"]),
        Case("nu_minus", b["c_eff"], 4.80, b["p_B"]),
        Case("nu_plus", b["c_eff"], 5.20, b["p_B"]),
        Case("p_B_minus", b["c_eff"], b["nu"], 0.28),
        Case("p_B_plus", b["c_eff"], b["nu"], 0.32),
    ]


def build_table_rows(case_rows: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
    base = case_rows["baseline"]
    specs = [
        ("c_eff", "c_eff_minus", "c_eff_plus", "0.45 / 0.50 / 0.55"),
        ("nu", "nu_minus", "nu_plus", "4.80 / 5.00 / 5.20"),
        ("p_B", "p_B_minus", "p_B_plus", "0.28 / 0.30 / 0.32"),
    ]

    out: List[Dict[str, str]] = []
    for pname, low_key, high_key, window in specs:
        low = case_rows[low_key]
        high = case_rows[high_key]
        out.append(
            {
                "parameter": pname,
                "window_low_base_high": window,
                "f_r3_gt_090_low_base_high": (
                    f"{low['f_r3_gt_090']:.4f} / {base['f_r3_gt_090']:.4f} / {high['f_r3_gt_090']:.4f}"
                ),
                "f_chi2_le_4_low_base_high": (
                    f"{low['f_chi2_le_4']:.4f} / {base['f_chi2_le_4']:.4f} / {high['f_chi2_le_4']:.4f}"
                ),
                "max_abs_drift_f_r3_gt_090": (
                    f"{max(abs(low['f_r3_gt_090'] - base['f_r3_gt_090']), abs(high['f_r3_gt_090'] - base['f_r3_gt_090'])):.4f}"
                ),
                "max_abs_drift_f_chi2_le_4": (
                    f"{max(abs(low['f_chi2_le_4'] - base['f_chi2_le_4']), abs(high['f_chi2_le_4'] - base['f_chi2_le_4'])):.4f}"
                ),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    d_knots, chi_knots = load_chi_knots()
    cases = build_cases()
    rows = [evaluate_case(c, d_knots, chi_knots) for c in cases]
    rows_by_name = {str(r["case"]): r for r in rows}
    table_rows = build_table_rows(rows_by_name)

    out_cases = OUTDIR / "core_param_robustness_cases.csv"
    out_table = OUTDIR / "core_param_robustness_table.csv"
    paper_table = PAPER_DIR / "core_param_robustness.csv"

    write_csv(out_cases, rows)
    write_csv(out_table, table_rows)
    write_csv(paper_table, table_rows)

    print(f"[saved] {out_cases}")
    print(f"[saved] {out_table}")
    print(f"[saved] {paper_table}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
