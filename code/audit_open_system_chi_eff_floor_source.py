#!/usr/bin/env python3
"""
Audit source reduction for the open-system chi_eff floor.

This script shows that on the exported exact micro bridge,

    chi_eff(D) = A_chi(D) * Cmax(D),

with

    A_chi(D) = 2 * gamma_mix(D) / gamma_ref(D)
             = kappa_env * 2 * g_x(D)^2 * Sxx(D) / gamma_ref(D),

so the residual observable-side floor problem can be reduced to a coherence
response floor on Cmax once A_chi is independently bounded from below.
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

    chi_eff = bridge["chi_eff_micro"].astype(float)
    cmax = bridge["Cmax_micro"].astype(float)
    gamma_mix = bridge["gamma_mix_micro"].astype(float)
    gamma_ref = bridge["gamma_ref"].astype(float)
    chi_lr = bridge["chi_LR_baseline"].astype(float)
    ratio = bridge["ratio_micro_to_baseline"].astype(float)
    g_x = bridge["g_x_micro"].astype(float)
    sxx = bridge["Sxx_delta"].astype(float)
    omega1 = bridge["omega1"].astype(float)
    kappa_env = bridge["kappa_env"].astype(float)

    m_lr_abs = (gamma_ref * chi_lr).abs()
    gamma_ref_from_omega11 = omega1**11

    a_chi_from_obs = chi_eff / cmax.clip(lower=1e-300)
    a_chi_from_rates = 2.0 * gamma_mix / gamma_ref.clip(lower=1e-300)
    a_chi_from_constituents = (
        kappa_env * 2.0 * (g_x**2) * sxx / gamma_ref.clip(lower=1e-300)
    )
    ratio_from_mlr = 2.0 * gamma_mix * cmax / m_lr_abs.clip(lower=1e-300)
    chi_eff_from_product = chi_lr * ratio

    bridge["M_LR_abs_from_baseline"] = m_lr_abs
    bridge["gamma_ref_from_omega11"] = gamma_ref_from_omega11
    bridge["A_chi_from_observable"] = a_chi_from_obs
    bridge["A_chi_from_rates"] = a_chi_from_rates
    bridge["A_chi_from_constituents"] = a_chi_from_constituents
    bridge["A_chi_obs_vs_rates_residual_abs"] = (a_chi_from_obs - a_chi_from_rates).abs()
    bridge["A_chi_obs_vs_constituents_residual_abs"] = (a_chi_from_obs - a_chi_from_constituents).abs()
    bridge["gamma_ref_omega11_residual_abs"] = (gamma_ref - gamma_ref_from_omega11).abs()
    bridge["ratio_from_mlr_recon"] = ratio_from_mlr
    bridge["ratio_from_mlr_residual_abs"] = (ratio - ratio_from_mlr).abs()
    bridge["chi_eff_from_product_recon"] = chi_eff_from_product
    bridge["chi_eff_from_product_residual_abs"] = (chi_eff - chi_eff_from_product).abs()

    gx_floor = float(g_x.min())
    gx_floor_D = float(bridge.loc[g_x.idxmin(), "D"])
    sxx_floor = float(sxx.min())
    sxx_floor_D = float(bridge.loc[sxx.idxmin(), "D"])
    omega1_ceil = float(omega1.max())
    omega1_ceil_D = float(bridge.loc[omega1.idxmax(), "D"])
    gamma_ref_ceil = float(omega1_ceil**11)
    kappa_const = float(kappa_env.iloc[0])

    a_chi_sep_floor = float(kappa_const * 2.0 * gx_floor * gx_floor * sxx_floor / max(gamma_ref_ceil, 1e-300))
    a_chi_exact_floor = float(a_chi_from_obs.min())
    a_chi_exact_floor_D = float(bridge.loc[a_chi_from_obs.idxmin(), "D"])

    cmax_floor = float(cmax.min())
    cmax_floor_D = float(bridge.loc[cmax.idxmin(), "D"])
    chi_eff_floor_from_sep_amp_exact_cmax = a_chi_sep_floor * cmax_floor
    chi_eff_exact_floor = float(chi_eff.min())
    chi_eff_exact_floor_D = float(bridge.loc[chi_eff.idxmin(), "D"])

    chi_lr_floor = float(chi_lr.min())
    chi_lr_floor_D = float(bridge.loc[chi_lr.idxmin(), "D"])
    ratio_floor = float(ratio.min())
    ratio_floor_D = float(bridge.loc[ratio.idxmin(), "D"])
    chi_eff_floor_from_exact_baseline_ratio = chi_lr_floor * ratio_floor

    out_detail = OUTDIR / "chi_open_system_chi_eff_floor_source_detail.csv"
    out_summary = OUTDIR / "chi_open_system_chi_eff_floor_source_summary.csv"
    bridge.to_csv(out_detail, index=False)

    summary = pd.DataFrame(
        [
            {
                "max_abs_A_chi_obs_vs_rates_residual": float(bridge["A_chi_obs_vs_rates_residual_abs"].max()),
                "max_abs_A_chi_obs_vs_constituents_residual": float(bridge["A_chi_obs_vs_constituents_residual_abs"].max()),
                "max_abs_gamma_ref_omega11_residual": float(bridge["gamma_ref_omega11_residual_abs"].max()),
                "max_abs_ratio_from_mlr_residual": float(bridge["ratio_from_mlr_residual_abs"].max()),
                "max_abs_chi_eff_product_residual": float(bridge["chi_eff_from_product_residual_abs"].max()),
                "kappa_env_const": kappa_const,
                "g_x_floor": gx_floor,
                "g_x_floor_D": gx_floor_D,
                "Sxx_floor": sxx_floor,
                "Sxx_floor_D": sxx_floor_D,
                "omega1_ceiling": omega1_ceil,
                "omega1_ceiling_D": omega1_ceil_D,
                "gamma_ref_ceiling_from_omega11": gamma_ref_ceil,
                "A_chi_separated_floor": a_chi_sep_floor,
                "A_chi_exact_floor": a_chi_exact_floor,
                "A_chi_exact_floor_D": a_chi_exact_floor_D,
                "actual_over_separated_A_chi_floor": a_chi_exact_floor / max(a_chi_sep_floor, 1e-300),
                "Cmax_exact_floor": cmax_floor,
                "Cmax_exact_floor_D": cmax_floor_D,
                "chi_eff_floor_from_separated_Achi_and_exact_Cmax_floor": chi_eff_floor_from_sep_amp_exact_cmax,
                "actual_chi_eff_floor": chi_eff_exact_floor,
                "actual_chi_eff_floor_D": chi_eff_exact_floor_D,
                "actual_over_AchiCmax_floor": chi_eff_exact_floor / max(chi_eff_floor_from_sep_amp_exact_cmax, 1e-300),
                "chi_LR_exact_floor": chi_lr_floor,
                "chi_LR_exact_floor_D": chi_lr_floor_D,
                "ratio_exact_floor": ratio_floor,
                "ratio_exact_floor_D": ratio_floor_D,
                "chi_eff_floor_from_exact_baseline_ratio_floors": chi_eff_floor_from_exact_baseline_ratio,
                "actual_over_exact_baseline_ratio_floor": chi_eff_exact_floor / max(chi_eff_floor_from_exact_baseline_ratio, 1e-300),
            }
        ]
    )
    summary.to_csv(out_summary, index=False)

    print(summary.to_string(index=False))
    print(f"[saved] {out_detail}")
    print(f"[saved] {out_summary}")


if __name__ == "__main__":
    main()
