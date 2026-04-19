#!/usr/bin/env python3
"""
Audit source for the Cmax ceiling in the open-system micro bridge.

Uses the exact micro-bridge export together with previously-audited numerator
residuals and gamma_ref / chi_eff floors to show that the universal two-level
positivity ceiling

    Cmax <= 1/2

already suffices to keep the dominant mix witness inside the audited half-radius
chart.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "chi_open_system"
PAPER_DIR = ROOT / "paper"


def read_csv_any(*paths: Path) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(paths[0])


def main() -> None:
    bridge = read_csv_any(
        OUTDIR / "chi_open_system_micro_bridge_map.csv",
        PAPER_DIR / "chi_open_system_micro_bridge_map.csv",
    ).copy()
    gamma_sum = read_csv_any(
        OUTDIR / "chi_open_system_gamma_mix_floor_source_summary.csv",
        PAPER_DIR / "chi_open_system_gamma_mix_floor_source_summary.csv",
    ).iloc[0]
    gamma_ref_sum = read_csv_any(
        OUTDIR / "chi_open_system_gamma_ref_floor_source_summary.csv",
        PAPER_DIR / "chi_open_system_gamma_ref_floor_source_summary.csv",
    ).iloc[0]

    bridge["half_ceiling_residual"] = 0.5 - bridge["Cmax_micro"].astype(float)
    bridge["positivity_quad_residual"] = 0.25 - bridge["Cmax_micro"].astype(float) ** 2

    gamma_ref_floor = float(gamma_ref_sum["actual_gamma_ref_floor"])
    chi_eff_floor = float(gamma_sum["min_chi_eff"])
    kappa_env_max = float(gamma_sum["kappa_env_max"])
    eps_mix = float(gamma_sum["eps_mix_factor_residual"])
    cmax_universal_ceiling = 0.5

    gamma_mix_floor_from_cmax_half = gamma_ref_floor * chi_eff_floor / (2.0 * cmax_universal_ceiling)
    kernel_floor_from_cmax_half = gamma_mix_floor_from_cmax_half / kappa_env_max
    mix_witness_bound_from_cmax_half = eps_mix / kernel_floor_from_cmax_half

    out_detail = OUTDIR / "chi_open_system_cmax_ceiling_source_detail.csv"
    out_summary = OUTDIR / "chi_open_system_cmax_ceiling_source_summary.csv"

    bridge.to_csv(out_detail, index=False)

    summary = pd.DataFrame(
        [
            {
                "max_Cmax": float(bridge["Cmax_micro"].max()),
                "max_Cmax_D": float(bridge.loc[bridge["Cmax_micro"].idxmax(), "D"]),
                "min_half_ceiling_residual": float(bridge["half_ceiling_residual"].min()),
                "min_positivity_quad_residual": float(bridge["positivity_quad_residual"].min()),
                "gamma_ref_floor": gamma_ref_floor,
                "chi_eff_floor": chi_eff_floor,
                "universal_Cmax_ceiling": cmax_universal_ceiling,
                "gamma_mix_floor_from_Cmax_half": gamma_mix_floor_from_cmax_half,
                "kernel_floor_from_Cmax_half": kernel_floor_from_cmax_half,
                "eps_mix_factor_residual": eps_mix,
                "mix_witness_bound_from_Cmax_half": mix_witness_bound_from_cmax_half,
                "p95_half_radius": float(gamma_sum["p95_half_radius"]),
                "max_half_radius": float(gamma_sum["max_half_radius"]),
                "p95_half_over_Cmax_half_bound": float(gamma_sum["p95_half_radius"]) / mix_witness_bound_from_cmax_half,
                "max_half_over_Cmax_half_bound": float(gamma_sum["max_half_radius"]) / mix_witness_bound_from_cmax_half,
            }
        ]
    )
    summary.to_csv(out_summary, index=False)

    print(summary.to_string(index=False))
    print(f"[saved] {out_detail}")
    print(f"[saved] {out_summary}")


if __name__ == "__main__":
    main()
