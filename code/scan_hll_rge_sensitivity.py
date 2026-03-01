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
from scan_hll_signal_strengths import PAPER_BASELINE, make_baseline_kinetics, snap_ref_d_for_full_direct


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "robustness"
PAPER_DIR = ROOT / "paper"

# Keep these aligned with scan_hll_signal_strengths.py parse_args() defaults
# so this sensitivity scan exercises the same release baseline runtime knobs.
RUNTIME_DIRECT_DEFAULTS = {
    "runtime_direct_force": False,
    "runtime_direct_no_cache": False,
    "runtime_direct_chi_rho_max": 3.0,
    "runtime_direct_chi_z_margin": 6.0,
    "runtime_direct_chi_n_mu": 120,
    "runtime_direct_chi_tol": 1e-8,
    "runtime_direct_chi_maxiter": 30000,
    "runtime_direct_chi_sigma": 2.5,
    "runtime_direct_superrad_zmax": 80.0,
    "runtime_direct_superrad_ref_d": 12.0,
    "runtime_direct_superrad_n_ref": 2,
}


@dataclass(frozen=True)
class Case:
    name: str
    uv_rge_mu_low: float
    uv_rge_gamma_diag: float
    uv_rge_gamma_offdiag: float
    uv_match_kappa_diag: float
    uv_match_kappa_offdiag: float


def evaluate_case(case: Case) -> Dict[str, float | str]:
    kin = make_baseline_kinetics(
        observable_mode="eft_wilson_uv_rge",
        chain_mode="full_direct",
        d_min=float(PAPER_BASELINE["D_min"]),
        d_max=float(PAPER_BASELINE["D_max"]),
        d_num=int(PAPER_BASELINE["D_num"]),
        uv_blend=float(PAPER_BASELINE["hll_uv_blend"]),
        uv_m2_power=float(PAPER_BASELINE["hll_uv_m2_power"]),
        uv_match_kappa_diag=float(case.uv_match_kappa_diag),
        uv_match_kappa_offdiag=float(case.uv_match_kappa_offdiag),
        uv_rge_mu_low=float(case.uv_rge_mu_low),
        uv_rge_gamma_diag=float(case.uv_rge_gamma_diag),
        uv_rge_gamma_offdiag=float(case.uv_rge_gamma_offdiag),
        uv_rge_log_clip=float(PAPER_BASELINE["hll_uv_rge_log_clip"]),
        runtime_direct_force=bool(RUNTIME_DIRECT_DEFAULTS["runtime_direct_force"]),
        runtime_direct_no_cache=bool(RUNTIME_DIRECT_DEFAULTS["runtime_direct_no_cache"]),
        runtime_direct_chi_rho_max=float(RUNTIME_DIRECT_DEFAULTS["runtime_direct_chi_rho_max"]),
        runtime_direct_chi_z_margin=float(RUNTIME_DIRECT_DEFAULTS["runtime_direct_chi_z_margin"]),
        runtime_direct_chi_n_mu=int(RUNTIME_DIRECT_DEFAULTS["runtime_direct_chi_n_mu"]),
        runtime_direct_chi_tol=float(RUNTIME_DIRECT_DEFAULTS["runtime_direct_chi_tol"]),
        runtime_direct_chi_maxiter=int(RUNTIME_DIRECT_DEFAULTS["runtime_direct_chi_maxiter"]),
        runtime_direct_chi_sigma=float(RUNTIME_DIRECT_DEFAULTS["runtime_direct_chi_sigma"]),
        runtime_direct_superrad_zmax=float(RUNTIME_DIRECT_DEFAULTS["runtime_direct_superrad_zmax"]),
        runtime_direct_superrad_ref_d=float(RUNTIME_DIRECT_DEFAULTS["runtime_direct_superrad_ref_d"]),
        runtime_direct_superrad_n_ref=int(RUNTIME_DIRECT_DEFAULTS["runtime_direct_superrad_n_ref"]),
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
    ref_d_eff, _ = snap_ref_d_for_full_direct("full_direct", float(PAPER_BASELINE["ref_D"]), d_vals)

    cfg = HLLObservableConfig(
        mode="eft_wilson_uv_rge",
        t_coh=float(PAPER_BASELINE["t_coh"]),
        ref_D=float(ref_d_eff),
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
        "uv_match_kappa_diag": float(case.uv_match_kappa_diag),
        "uv_match_kappa_offdiag": float(case.uv_match_kappa_offdiag),
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
    kdiag0 = float(PAPER_BASELINE["hll_uv_match_kappa_diag"])
    koff0 = float(PAPER_BASELINE["hll_uv_match_kappa_offdiag"])
    return [
        Case("baseline", 1.0, 2.0, 1.0, kdiag0, koff0),
        Case("mu_low_minus", 0.5, 2.0, 1.0, kdiag0, koff0),
        Case("mu_low_plus", 2.0, 2.0, 1.0, kdiag0, koff0),
        Case("gamma_diag_minus", 1.0, 1.0, 1.0, kdiag0, koff0),
        Case("gamma_diag_plus", 1.0, 3.0, 1.0, kdiag0, koff0),
        Case("gamma_offdiag_minus", 1.0, 2.0, 0.5, kdiag0, koff0),
        Case("gamma_offdiag_plus", 1.0, 2.0, 1.5, kdiag0, koff0),
        Case("kappa_diag_minus", 1.0, 2.0, 1.0, -1.0, koff0),
        Case("kappa_diag_plus", 1.0, 2.0, 1.0, +1.0, koff0),
        Case("kappa_offdiag_minus", 1.0, 2.0, 1.0, kdiag0, -1.0),
        Case("kappa_offdiag_plus", 1.0, 2.0, 1.0, kdiag0, +1.0),
    ]


def build_table_rows(rows_by_name: Dict[str, Dict[str, float | str]]) -> List[Dict[str, str]]:
    base = rows_by_name["baseline"]
    specs = [
        ("mu_low", "mu_low_minus", "mu_low_plus", "0.5 / 1.0 / 2.0"),
        ("gamma_diag", "gamma_diag_minus", "baseline", "1.0 / 2.0 / 3.0", "gamma_diag_plus"),
        ("gamma_offdiag", "gamma_offdiag_minus", "baseline", "0.5 / 1.0 / 1.5", "gamma_offdiag_plus"),
        ("kappa_diag", "kappa_diag_minus", "baseline", "-1.0 / 0.0 / 1.0", "kappa_diag_plus"),
        ("kappa_offdiag", "kappa_offdiag_minus", "baseline", "-1.0 / 0.0 / 1.0", "kappa_offdiag_plus"),
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
