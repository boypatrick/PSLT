#!/usr/bin/env python3
"""
Audit the remaining amplitude-side slack in A_chi on the exact open-system bridge.

This script shows that the exact amplitude floor is aligned with the gamma_mix
floor knot on the exported bridge:

    A_chi(D) = 2 gamma_mix(D) / omega1(D)^11

and the remaining slack in the separated floor is entirely the penalty from
replacing the local denominator omega1(D_gamma_min)^11 by the global ceiling
omega1_max^11.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / 'output' / 'chi_open_system'
PAPER_DIR = ROOT / 'paper'


def read_csv_any(*paths: Path) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(paths[0])


def main() -> None:
    bridge = read_csv_any(
        OUTDIR / 'chi_open_system_micro_bridge_map.csv',
        PAPER_DIR / 'chi_open_system_micro_bridge_map.csv',
    ).copy()
    ratio_sum = read_csv_any(
        OUTDIR / 'chi_open_system_cmax_ratio_floor_source_summary.csv',
        PAPER_DIR / 'chi_open_system_cmax_ratio_floor_source_summary.csv',
    ).iloc[0]

    D = bridge['D'].astype(float)
    kappa_env = bridge['kappa_env'].astype(float)
    omega1 = bridge['omega1'].astype(float)
    gamma_mix = bridge['gamma_mix_micro'].astype(float)
    gx = bridge['g_x_micro'].astype(float)
    sxx = bridge['Sxx_delta'].astype(float)
    chi_eff = bridge['chi_eff_micro'].astype(float)
    cmax = bridge['Cmax_micro'].astype(float)

    A_obs = chi_eff / cmax
    A_from_rates = 2.0 * gamma_mix / np.power(omega1, 11)
    A_from_kernel = 2.0 * kappa_env * (gx * gx * sxx) / np.power(omega1, 11)
    kernel_gap_ratio = (gx * gx * sxx) / np.power(omega1, 11)

    bridge['A_chi_obs'] = A_obs
    bridge['A_chi_from_rates'] = A_from_rates
    bridge['A_chi_from_kernel_gap_ratio'] = A_from_kernel
    bridge['kernel_gap_ratio'] = kernel_gap_ratio
    bridge['A_chi_obs_vs_rates_residual_abs'] = np.abs(A_obs - A_from_rates)
    bridge['A_chi_obs_vs_kernel_residual_abs'] = np.abs(A_obs - A_from_kernel)

    idx_A = int(np.argmin(A_obs))
    idx_gm = int(np.argmin(gamma_mix))
    idx_ratio = int(np.argmin(kernel_gap_ratio))

    A_exact_floor = float(A_obs.iloc[idx_A])
    D_A = float(D.iloc[idx_A])
    gamma_mix_floor = float(gamma_mix.iloc[idx_gm])
    D_gm = float(D.iloc[idx_gm])
    kernel_gap_ratio_floor = float(kernel_gap_ratio.iloc[idx_ratio])
    D_ratio = float(D.iloc[idx_ratio])

    omega_at_gamma_min = float(omega1.iloc[idx_gm])
    omega_ceil = float(omega1.max())

    A_from_gamma_local = 2.0 * gamma_mix_floor / (omega_at_gamma_min ** 11)
    A_from_gamma_global_omega_ceil = 2.0 * gamma_mix_floor / (omega_ceil ** 11)
    exact_over_gamma_global = A_exact_floor / max(A_from_gamma_global_omega_ceil, 1e-300)

    cmax_ratio_rigorous_floor = float(ratio_sum['Cmax_ratio_rigorous_floor'])
    chi_eff_floor_from_exact_A_and_ratio_C = A_exact_floor * cmax_ratio_rigorous_floor
    actual_chi_eff_floor = float(chi_eff.min())

    out_detail = OUTDIR / 'chi_open_system_achi_amplitude_source_detail.csv'
    out_summary = OUTDIR / 'chi_open_system_achi_amplitude_source_summary.csv'
    bridge.to_csv(out_detail, index=False)

    summary = pd.DataFrame([{
        'n_rows': int(len(bridge)),
        'max_abs_A_obs_vs_rates_residual': float(bridge['A_chi_obs_vs_rates_residual_abs'].max()),
        'max_abs_A_obs_vs_kernel_residual': float(bridge['A_chi_obs_vs_kernel_residual_abs'].max()),
        'A_chi_exact_floor': A_exact_floor,
        'A_chi_exact_floor_D': D_A,
        'gamma_mix_floor': gamma_mix_floor,
        'gamma_mix_floor_D': D_gm,
        'kernel_gap_ratio_floor': kernel_gap_ratio_floor,
        'kernel_gap_ratio_floor_D': D_ratio,
        'all_argmins_align': int(idx_A == idx_gm == idx_ratio),
        'omega1_at_gamma_mix_floor': omega_at_gamma_min,
        'omega1_ceil': omega_ceil,
        'A_chi_from_gamma_local_floor': A_from_gamma_local,
        'A_chi_from_gamma_global_omega_ceil': A_from_gamma_global_omega_ceil,
        'exact_over_gamma_global_omega_ceil_floor': exact_over_gamma_global,
        'Cmax_ratio_rigorous_floor': cmax_ratio_rigorous_floor,
        'chi_eff_floor_from_exact_A_and_ratio_C': chi_eff_floor_from_exact_A_and_ratio_C,
        'actual_chi_eff_floor': actual_chi_eff_floor,
        'actual_over_exactA_ratioC_floor': actual_chi_eff_floor / max(chi_eff_floor_from_exact_A_and_ratio_C, 1e-300),
    }])
    summary.to_csv(out_summary, index=False)

    print(summary.to_string(index=False))
    print(f'[saved] {out_detail}')
    print(f'[saved] {out_summary}')


if __name__ == '__main__':
    main()
