#!/usr/bin/env python3
"""
Select dynamic H->ll reference anchors from a fixed-reference scan.

Outputs:
  - output/hll_reference_anchor/reference_anchor_candidates.csv
  - output/hll_reference_anchor/reference_anchor_choice.json
  - paper/reference_anchor_candidates.csv
  - paper/reference_anchor_choice.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from hll_observable import HLLObservableConfig, HLLChannelPredictor
from pslt_lib import PSLTKinetics, PSLTParameters
from reference_anchor_utils import select_anchor_candidates_from_fixed_scan


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "hll_reference_anchor"
PAPER_DIR = ROOT / "paper"
CHI_CSV = ROOT / "output" / "chi_fp_2d" / "localized_chi_D6-12-18.csv"
B_OVERLAP_CSV = ROOT / "output" / "y_eff_2d" / "y_eff_2d_three_channel_profile.csv"

BASELINE = {
    "c_eff": 0.5,
    "nu": 5.0,
    "kappa_g": 0.03,
    "g_mode": "fp_2d_full",
    "g_fp_norm_mode": "phase_space",
    "g_fp_full_window_blend": 0.8,
    "g_fp_full_tail_beta": 1.1,
    "g_fp_full_tail_shell_power": 0.0,
    "g_fp_full_tail_clip_min": 1e-3,
    "g_fp_full_tail_clip_max": 0.95,
    "chi_legacy": 0.2,
    "chi_mode": "localized_interp",
    "A1": 1.0,
    "A2": 1.0,
    "gamma_mode": "action_profile",
    "p_B": 0.30,
    "b_mode": "overlap_2d",
    "hll_observable_mode": "eft_wilson_matched",
    "hll_observable_nmax": 20,
    "t_coh": 1.0,
    "mu_obs": 1.4,
    "sigma_obs": 0.4,
    "D_min": 4.0,
    "D_max": 20.0,
    "D_num": 60,
    "eta_min": 0.2,
    "eta_max": 4.0,
    "eta_num": 60,
}


def load_chi_knots(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing chi profile csv: {path}")
    rows: list[Tuple[float, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("level", "").strip().lower() == "fine":
                rows.append((float(row["D"]), float(row["chi_LR"])))
    if len(rows) < 2:
        raise RuntimeError(f"Need >=2 fine rows in {path}")
    rows.sort(key=lambda t: t[0])
    d = np.array([x for x, _ in rows], dtype=float)
    chi = np.array([y for _, y in rows], dtype=float)
    return d, chi


def make_kinetics(d_knots: np.ndarray, chi_knots: np.ndarray) -> PSLTKinetics:
    params = PSLTParameters(
        c_eff=BASELINE["c_eff"],
        nu=BASELINE["nu"],
        kappa_g=BASELINE["kappa_g"],
        g_mode=BASELINE["g_mode"],
        g_fp_norm_mode=BASELINE["g_fp_norm_mode"],
        g_fp_full_window_blend=BASELINE["g_fp_full_window_blend"],
        g_fp_full_tail_beta=BASELINE["g_fp_full_tail_beta"],
        g_fp_full_tail_shell_power=BASELINE["g_fp_full_tail_shell_power"],
        g_fp_full_tail_clip_min=BASELINE["g_fp_full_tail_clip_min"],
        g_fp_full_tail_clip_max=BASELINE["g_fp_full_tail_clip_max"],
        chi=BASELINE["chi_legacy"],
        chi_mode=BASELINE["chi_mode"],
        chi_lr_D=tuple(float(x) for x in d_knots),
        chi_lr_vals=tuple(float(y) for y in chi_knots),
        A1=BASELINE["A1"],
        A2=BASELINE["A2"],
        gamma_mode=BASELINE["gamma_mode"],
        b_mode=BASELINE["b_mode"],
        b_overlap_csv=str(B_OVERLAP_CSV),
        b_n_power=BASELINE["p_B"],
        b_n_mode="cumulative",
        b_n_tail_mode="saturate",
        hll_observable_mode=BASELINE["hll_observable_mode"],
        hll_observable_nmax=BASELINE["hll_observable_nmax"],
    )
    return PSLTKinetics(params)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select dynamic reference anchors from fixed-reference scan.")
    ap.add_argument("--fixed-ref-d", type=float, default=10.0)
    ap.add_argument("--fixed-ref-eta", type=float, default=1.0)
    ap.add_argument("--recommended-mode", choices=["fixed", "chi2_best", "robust_center"], default="robust_center")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    d_knots, chi_knots = load_chi_knots(CHI_CSV)
    kinetics = make_kinetics(d_knots, chi_knots)

    d_vals = np.linspace(BASELINE["D_min"], BASELINE["D_max"], BASELINE["D_num"])
    eta_vals = np.linspace(BASELINE["eta_min"], BASELINE["eta_max"], BASELINE["eta_num"])

    candidates = select_anchor_candidates_from_fixed_scan(
        kinetics=kinetics,
        d_vals=d_vals,
        eta_vals=eta_vals,
        mode=BASELINE["hll_observable_mode"],
        t_coh=float(BASELINE["t_coh"]),
        n_max=int(BASELINE["hll_observable_nmax"]),
        mu_obs=float(BASELINE["mu_obs"]),
        sigma_obs=float(BASELINE["sigma_obs"]),
        fixed_ref_d=float(args.fixed_ref_d),
        fixed_ref_eta=float(args.fixed_ref_eta),
    )

    # Add a quick acceptance metric under each candidate as reference.
    enriched_rows: list[Dict[str, float | str]] = []
    for key in ("fixed", "chi2_best", "robust_center"):
        row = dict(candidates[key])
        cfg = HLLObservableConfig(
            mode=BASELINE["hll_observable_mode"],
            t_coh=float(BASELINE["t_coh"]),
            ref_D=float(row["ref_D"]),
            ref_eta=float(row["ref_eta"]),
            n_max=int(BASELINE["hll_observable_nmax"]),
        )
        predictor = HLLChannelPredictor(kinetics, layer_n=2, cfg=cfg)
        chi2 = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
        for i, eta in enumerate(eta_vals):
            for j, d in enumerate(d_vals):
                mu_pred = float(predictor.mu_pred(float(d), float(eta)))
                chi2[i, j] = ((mu_pred - BASELINE["mu_obs"]) / BASELINE["sigma_obs"]) ** 2
        row["f_chi2_le_4_reanchored"] = float(np.mean(chi2 <= 4.0))
        row["best_chi2_reanchored"] = float(np.min(chi2))
        enriched_rows.append(row)

    out_csv = OUTDIR / "reference_anchor_candidates.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(enriched_rows[0].keys()))
        writer.writeheader()
        writer.writerows(enriched_rows)

    selected_by_mode = {
        key: {"D": float(candidates[key]["ref_D"]), "eta": float(candidates[key]["ref_eta"])}
        for key in ("fixed", "chi2_best", "robust_center")
    }
    rec_mode = str(args.recommended_mode)
    out_json = OUTDIR / "reference_anchor_choice.json"
    payload = {
        "hll_observable_mode": BASELINE["hll_observable_mode"],
        "selection_from_fixed_scan": candidates["meta"],
        "selected_by_mode": selected_by_mode,
        "recommended_mode": rec_mode,
        "recommended": selected_by_mode[rec_mode],
    }
    out_json.write_text(json.dumps(payload, indent=2))

    paper_csv = PAPER_DIR / out_csv.name
    paper_json = PAPER_DIR / out_json.name
    paper_csv.write_text(out_csv.read_text())
    paper_json.write_text(out_json.read_text())

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")
    print(f"[saved] {paper_csv}")
    print(f"[saved] {paper_json}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

