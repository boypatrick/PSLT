#!/usr/bin/env python3
"""
One-at-a-time sensitivity scan for LL-RG parameters in the UV->EFT H->mumu map.

Outputs:
  - output/robustness/hll_rge_sensitivity_cases.csv
  - output/robustness/hll_rge_sensitivity_table.csv
  - paper/hll_rge_sensitivity.csv
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from hll_observable import HLLObservableConfig, HLLChannelPredictor
from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "robustness"
PAPER_DIR = ROOT / "paper"


@dataclass(frozen=True)
class Case:
    name: str
    uv_rge_mu_low: float
    uv_rge_gamma_diag: float
    uv_rge_gamma_offdiag: float


def evaluate_case(case: Case) -> Dict[str, float | str]:
    kin = make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        uv_blend=float(PAPER_BASELINE["hll_uv_blend"]),
        uv_m2_power=float(PAPER_BASELINE["hll_uv_m2_power"]),
        uv_rge_mu_low=float(case.uv_rge_mu_low),
        uv_rge_gamma_diag=float(case.uv_rge_gamma_diag),
        uv_rge_gamma_offdiag=float(case.uv_rge_gamma_offdiag),
        uv_rge_log_clip=float(PAPER_BASELINE["hll_uv_rge_log_clip"]),
    )

    d_vals = np.linspace(
        float(PAPER_BASELINE["D_min"]),
        float(PAPER_BASELINE["D_max"]),
        int(PAPER_BASELINE["D_num"]),
    )
    eta_vals = np.linspace(
        float(PAPER_BASELINE["eta_min"]),
        float(PAPER_BASELINE["eta_max"]),
        int(PAPER_BASELINE["eta_num"]),
    )

    cfg = HLLObservableConfig(
        mode="eft_wilson_uv_rge",
        t_coh=float(PAPER_BASELINE["t_coh"]),
        ref_D=float(PAPER_BASELINE["ref_D"]),
        ref_eta=float(PAPER_BASELINE["ref_eta"]),
        n_max=int(PAPER_BASELINE["hll_observable_nmax"]),
    )
    pred = HLLChannelPredictor(kin, layer_n=2, cfg=cfg)

    mu = np.zeros((len(eta_vals), len(d_vals)), dtype=float)
    chi2 = np.zeros_like(mu)
    mu_obs = 1.4
    sigma_obs = 0.4

    for i, eta in enumerate(eta_vals):
        for j, d in enumerate(d_vals):
            m = float(pred.mu_pred(float(d), float(eta)))
            mu[i, j] = m
            chi2[i, j] = ((m - mu_obs) / sigma_obs) ** 2

    best_idx = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
    return {
        "case": case.name,
        "uv_rge_mu_low": float(case.uv_rge_mu_low),
        "uv_rge_gamma_diag": float(case.uv_rge_gamma_diag),
        "uv_rge_gamma_offdiag": float(case.uv_rge_gamma_offdiag),
        "f_chi2_le_4": float(np.mean(chi2 <= 4.0)),
        "best_chi2": float(np.min(chi2)),
        "best_D": float(d_vals[int(best_idx[1])]),
        "best_eta": float(eta_vals[int(best_idx[0])]),
        "mu_p50": float(np.percentile(mu, 50.0)),
        "mu_p90": float(np.percentile(mu, 90.0)),
        "mu_mean": float(np.mean(mu)),
        "mu_std": float(np.std(mu)),
    }


def build_cases() -> List[Case]:
    return [
        Case("baseline", 1.0, 2.0, 1.0),
        Case("mu_low_minus", 0.5, 2.0, 1.0),
        Case("mu_low_plus", 2.0, 2.0, 1.0),
        Case("gamma_diag_minus", 1.0, 1.0, 1.0),
        Case("gamma_diag_plus", 1.0, 3.0, 1.0),
        Case("gamma_offdiag_minus", 1.0, 2.0, 0.5),
        Case("gamma_offdiag_plus", 1.0, 2.0, 1.5),
    ]


def build_table_rows(rows_by_name: Dict[str, Dict[str, float | str]]) -> List[Dict[str, str]]:
    base = rows_by_name["baseline"]
    specs = [
        ("mu_low", "mu_low_minus", "mu_low_plus", "0.5 / 1.0 / 2.0"),
        ("gamma_diag", "gamma_diag_minus", "baseline", "1.0 / 2.0 / 3.0", "gamma_diag_plus"),
        ("gamma_offdiag", "gamma_offdiag_minus", "baseline", "0.5 / 1.0 / 1.5", "gamma_offdiag_plus"),
    ]

    out: List[Dict[str, str]] = []
    for spec in specs:
        if spec[0] == "mu_low":
            pname, low_key, high_key, window = spec
            low = rows_by_name[low_key]
            high = rows_by_name[high_key]
        else:
            pname, low_key, base_key, window, high_key = spec
            low = rows_by_name[low_key]
            high = rows_by_name[high_key]

        out.append(
            {
                "parameter": str(spec[0]),
                "window_low_base_high": str(window),
                "f_chi2_le_4_low_base_high": (
                    f"{float(low['f_chi2_le_4']):.4f} / {float(base['f_chi2_le_4']):.4f} / {float(high['f_chi2_le_4']):.4f}"
                ),
                "best_chi2_low_base_high": (
                    f"{float(low['best_chi2']):.3e} / {float(base['best_chi2']):.3e} / {float(high['best_chi2']):.3e}"
                ),
                "max_abs_drift_f_chi2_le_4": (
                    f"{max(abs(float(low['f_chi2_le_4']) - float(base['f_chi2_le_4'])), abs(float(high['f_chi2_le_4']) - float(base['f_chi2_le_4']))):.4f}"
                ),
            }
        )

    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    cases = build_cases()
    rows = [evaluate_case(c) for c in cases]
    rows_by_name = {str(r["case"]): r for r in rows}
    table_rows = build_table_rows(rows_by_name)

    out_cases = OUTDIR / "hll_rge_sensitivity_cases.csv"
    out_table = OUTDIR / "hll_rge_sensitivity_table.csv"
    paper_csv = PAPER_DIR / "hll_rge_sensitivity.csv"

    write_csv(out_cases, rows)
    write_csv(out_table, table_rows)
    write_csv(paper_csv, table_rows)

    print(f"[saved] {out_cases}")
    print(f"[saved] {out_table}")
    print(f"[saved] {paper_csv}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
